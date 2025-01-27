# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time

import backoff
import openai
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
)
import requests
from dotenv import load_dotenv
import litellm
import base64

EMPTY_API_KEY="Your API KEY Here"

def load_openai_api_key():
    load_dotenv()
    assert (
            os.getenv("OPENAI_API_KEY") is not None and
            os.getenv("OPENAI_API_KEY") != EMPTY_API_KEY
    ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
    return os.getenv("OPENAI_API_KEY")


def load_gemini_api_key():
    load_dotenv()
    assert (
            os.getenv("GEMINI_API_KEY") is not None and
            os.getenv("GEMINI_API_KEY") != EMPTY_API_KEY
    ), "must pass on the api_key or set GEMINI_API_KEY in the environment"
    return os.getenv("GEMINI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def engine_factory(api_key=None, model=None, **kwargs):
    model = model.lower()
    if kwargs.get("grounding_model_config") is not None:
        grounding_model_config = kwargs.pop("grounding_model_config")
        default_model_config = {'api_key': api_key, 'model': model}
        logger = kwargs.pop("logger")
        default_model_config.update(kwargs)
        return BiOpenAIEngine(default_model_config=default_model_config, grounding_model_config=grounding_model_config, logger=logger, **kwargs)
    if model in ["gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
        if api_key and api_key != EMPTY_API_KEY:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            load_openai_api_key()
        return OpenAIEngine(model=model, **kwargs)
    elif model in ["gemini-1.5-pro-latest", "gemini-1.5-flash"]:
        if api_key and api_key != EMPTY_API_KEY:
            os.environ["GEMINI_API_KEY"] = api_key
        else:
            load_gemini_api_key()
        model=f"gemini/{model}"
        return GeminiEngine(model=model, **kwargs)
    elif model == "llava":
        model="llava"
        return OllamaEngine(model=model, **kwargs)
    raise Exception(f"Unsupported model: {model}, currently supported models: \
                    gpt-4-vision-preview, gpt-4-turbo, gpt-4o, , gpt-4o-mini, gemini-1.5-pro-latest, llava")

class Engine:
    def __init__(
            self,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """
            Base class to init an engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        self.time_slots = [0]
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.time_slots)
        self.current_key_idx = 0
        print(f"Initializing model {self.model}")        

    def tokenize(self, input):
        return self.tokenizer(input)


class OllamaEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init an Ollama engine
            To use Ollama, dowload and install Ollama from https://ollama.com/
            After Ollama start, pull llava with command: ollama pull llava
        """
        super().__init__(**kwargs)
        self.api_url = "http://localhost:11434/api/chat"


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
            time.sleep(wait_time)
        prompt0, prompt1, prompt2 = prompt

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
                {"role": "assistant", "content": f"\n\n{ouput_0}"},
                {"role": "user", "content": prompt2}, 
            ]

        options = {"temperature": self.temperature, "num_predict": max_new_tokens}
        data = {
            "model": self.model,
            "messages": prompt_input,
            "options": options,
            "stream": False,
        }
        _request = {
            "url": f"{self.api_url}",
            "json": data,
        }
        response = requests.post(**_request)  # type: ignore
        if response.status_code != 200:
            raise Exception(f"Ollama API Error: {response.status_code}, {response.text}")
        response_json = response.json()
        return response_json["message"]["content"]


class GeminiEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init a Gemini engine
            To use this engine, please provide the GEMINI_API_KEY in the environment
            Supported Model             Rate Limit
            gemini-1.5-pro-latest    	2 queries per minute, 1000 queries per day
        """
        super().__init__(**kwargs)


    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            wait_time = self.next_avil_time[self.current_key_idx] - start_time
            print(f"Wait {wait_time} for rate limitting")
        prompt0, prompt1, prompt2 = prompt
        litellm.set_verbose=True

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"},
                                                                }]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "system", "content": prompt0},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": image_path,
                                                                                                    "detail": "high"}, 
                                                                }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature if temperature else self.temperature,
            **kwargs,
        )
        return [choice["message"]["content"] for choice in response.choices][0]


class OpenAIEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
            Init an OpenAI GPT/Codex engine
            To find your OpenAI API key, visit https://platform.openai.com/api-keys
        """
        super().__init__(**kwargs)

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        prompt0, prompt1, prompt2 = prompt
        # litellm.set_verbose=True

        base64_image = encode_image(image_path)
        if turn_number == 0:
            # Assume one turn dialogue
            prompt_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "high"},
                                                                 }]},
            ]
        elif turn_number == 1:
            prompt_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url":
                                                                                                        f"data:image/jpeg;base64,{base64_image}",
                                                                                                    "detail": "high"}, }]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temperature if temperature else self.temperature,
            **kwargs,
        )
        return [choice["message"]["content"] for choice in response.choices][0]
    

from openai import OpenAI
from PIL import Image, ImageDraw
import re
from copy import deepcopy
class BiOpenAIEngine(Engine):
    def __init__(self, default_model_config, grounding_model_config, logger, **kwargs) -> None:
        # super().__init__(stop, rate_limit, model, temperature, **kwargs)
        self.logger = logger
        default_model_config = deepcopy(default_model_config)
        grounding_model_config = deepcopy(grounding_model_config)
        self.model_names = {
            "grounding": grounding_model_config.pop("model"),
            "default": default_model_config.pop("model"),
        }
        self.temperature = default_model_config.pop("temperature", 0.0)
        default_model_config.pop("rate_limit", None)
        grounding_model_config.pop("rate_limit", None)
        self.clients: dict[str, OpenAI] = {
            "grounding": OpenAI(**grounding_model_config),
            "default": OpenAI(**default_model_config),
        }

    def _init_one_client(self, model_config):
        return 

    def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, **kwargs):
        
        # prompt0, prompt1, prompt2 = prompt
        client = self.clients["default"]
        model_name = self.model_names["default"]
        base64_image = encode_image(image_path)

        kwargs.update({
            "max_tokens": max_new_tokens if max_new_tokens else 4096,
            "temperature": temperature if temperature else self.temperature,
        })

        if turn_number == 0:
            # Assume one turn dialogue
            prompt0, prompt1 = prompt
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                ]},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_input,
                **kwargs,
            )
        elif turn_number == 1:
            prompt0, prompt1, prompt2 = prompt
            prompt_input = [
                {"role": "assistant", "content": prompt0},
                # {"role": "user", "content": prompt1, "images": [f"{base64_image}"]},
                {
                    "role": "user", "content": [
                        {"type": "text", "text": prompt1},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                    ]
                },
                {"role": "assistant", "content": f"\n\n{ouput_0}"},
                {"role": "user", "content": prompt2}, 
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_input,
                **kwargs,
            )
        return response.choices[0].message.content
    
    def _extract_coordinates(self, text):
        match = re.search(r'(\d+)[^\d]+(\d+)', text)
        if match:
            return {"x": int(match.group(1)), "y": int(match.group(2))}
        return None
    
    def grounding_generate(self, prompt: list = str, image_path=None, max_new_tokens=4096, temperature=None):
        client = self.clients["grounding"]
        model_name = self.model_names["grounding"]
        image = Image.open(image_path)
        image_width, image_height = image.size
        base64_image = encode_image(image_path)
        kwargs = {
            "max_tokens": max_new_tokens if max_new_tokens else 4096,
            "temperature": temperature if temperature else self.temperature,
        }
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs,
        )

        # Open the screenshot image
        coordinates_text = response.choices[0].message.content

        coordinates_ratio = self._extract_coordinates(coordinates_text)
        coordinates = {
            "x": round(coordinates_ratio["x"] * image_width / 1000),
            "y": round(coordinates_ratio["y"] * image_height / 1000),
        }
        if coordinates:

            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Draw a green dot at the click coordinates
            radius = 6
            x, y = coordinates["x"], coordinates["y"]
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="green")

            # Save the modified image
            click_screenshot_path = image_path.replace('_full.png', '_grounding.png')
            image.save(click_screenshot_path)
            self.logger.info(f"Grounding coordinates screenshot to: {click_screenshot_path}")
        else:
            self.logger.warning(f"Failed to extract coordinates from: {coordinates_text}")

        return coordinates_text, coordinates

class OpenaiEngine_MindAct(Engine):
    def __init__(self, **kwargs) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        super().__init__(**kwargs)
    #
    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, prompt, max_new_tokens=50, temperature=0, model=None, **kwargs):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        if isinstance(prompt, str):
            # Assume one turn dialogue
            prompt = [
                {"role": "user", "content": prompt},
            ]
        response = litellm.completion(
            model=model if model else self.model,
            messages=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                    max(start_time, self.next_avil_time[self.current_key_idx])
                    + self.request_interval
            )
        return [choice["message"]["content"] for choice in response["choices"]]
