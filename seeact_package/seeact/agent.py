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

import json
import logging
import os
import random
import traceback
import re
from datetime import datetime
from os.path import dirname

import toml
import base64
from playwright.async_api import async_playwright, Locator
from playwright._impl._page import Page


from .data_utils.format_prompt_utils import get_index_from_option_name, generate_new_query_prompt, \
    generate_new_referring_prompt, format_options, generate_option_name
from .demo_utils.browser_helper import normal_launch_async, normal_new_context_async, \
    get_interactive_elements_with_playwright, get_selectors_with_playwright, select_option, saveconfig
from .demo_utils.crawler_helper import get_random_link
from .demo_utils.format_prompt import format_choices, postprocess_action_lmm, postprocess_action_lmm_pixel
from .demo_utils.inference_engine import engine_factory
from PIL import Image, ImageDraw
import io
import difflib

def locator_serializer(obj):
    """Convert non-serializable objects to a serializable format."""
    if isinstance(obj, Locator):
        # Assuming Locator has attributes 'frame' and 'selector' you want to serialize
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class SeeActAgent:
    def __init__(self,
                 config_path=None,
                 save_file_dir="seeact_agent_files",
                 default_task='Find the pdf of the paper "GPT-4V(ision) is a Generalist Web Agent, if Grounded"',
                 default_website="https://www.google.com/",
                 input_info=["screenshot"],
                 grounding_strategy="text_choice_som",  # [...,'pixel_2_stage']
                 crawler_mode=False,
                 crawler_max_steps=10,
                 max_auto_op=50,
                 max_continuous_no_op=5,
                 highlight=False,
                 headless=False,
                 args=[],
                 browser_app="chrome",
                 persistant=False,
                 persistant_user_path="",
                 save_video=False,
                 viewport={
                     "width": 1280,
                     "height": 720
                 },
                 tracing=False,
                 trace={
                     "screenshots": True,
                     "snapshots": True,
                     "sources": True
                 },
                 rate_limit=-1,
                 model="gpt-4o",
                 temperature=0.9,
                 grounding_model_config=None,
                 ):

        try:
            if config_path is not None:
                with open(config_path,
                          'r') as config:
                    print(f"Configuration File Loaded - {config_path}")
                    config = toml.load(config)
            else:
                config = {
                    "basic": {
                        "save_file_dir": save_file_dir,
                        "default_task": default_task,
                        "default_website": default_website,
                        "crawler_mode": crawler_mode,
                        "crawler_max_steps": crawler_max_steps,
                    },
                    "agent": {
                        "input_info": input_info,
                        "grounding_strategy": grounding_strategy,
                        "max_auto_op": max_auto_op,
                        "max_continuous_no_op": max_continuous_no_op,
                        "highlight": highlight,
                    },
                    "openai": {
                        "rate_limit": rate_limit,
                        "model": model,
                        "temperature": temperature
                    }
                }
            config.update({
                "browser": {
                    "headless": headless,
                    "args": args,
                    "browser_app": browser_app,
                    "persistant": persistant,
                    "persistant_user_path": persistant_user_path,
                    "save_video": save_video,
                    "viewport": viewport,
                    "tracing": tracing,
                    "trace": trace
                }
            })

        except FileNotFoundError:
            print(f"Error: File '{os.path.abspath(config_path)}' not found.")
        except toml.TomlDecodeError:
            print(f"Error: File '{os.path.abspath(config_path)}' is not a valid TOML file.")

        self.config = config
        self.complete_flag = False
        self.session_control = {
            'active_page': None,
            'context': None,
            'browser': None
        }
        self.tasks = [self.config["basic"]["default_task"]]

        self.main_path = os.path.join(self.config["basic"]["save_file_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.main_path, exist_ok=True)
        self.action_space = ["CLICK", "PRESS ENTER", "HOVER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB",
                             "GO BACK", "GO FORWARD",
                             "TERMINATE", "SELECT", "TYPE", "GOTO", "MEMORIZE"]  # Define the list of actions here

        self.no_value_op = ["CLICK", "PRESS ENTER", "HOVER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB",
                            "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN"
                                                                       "GO BACK",
                            "GO FORWARD",
                            "TERMINATE", "NONE"]

        self.with_value_op = ["SELECT", "TYPE", "GOTO", "MEMORIZE", "SAY"]

        self.no_element_op = ["PRESS ENTER", "SCROLL UP", "SCROLL DOWN", "NEW TAB", "CLOSE TAB", "GO BACK", "GOTO",
                              "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                              "GO FORWARD",
                              "TERMINATE", "NONE", "MEMORIZE", "SAY"]

        # Initialize the primary logger and the developer logger
        self.logger = self._setup_logger(redirect_to_dev_log=False)
        # self.dev_logger = self._setup_dev_logger()

        # # Redirect primary logger messages to dev_logger as well
        # for handler in self.logger.handlers:
        #     self.dev_logger.addHandler(handler)
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage" and grounding_model_config is None:
            grounding_model_config = self.config["openai"]
        self.engine = engine_factory(grounding_model_config=grounding_model_config, logger=self.logger, **self.config['openai'])
        self.taken_actions = []

        # if self.config["agent"]["grounding_strategy"] == "pixel_2_stage":
        self.prompts = self._initialize_prompts_pure_vision()
        self.time_step = 0
        self.valid_op = 0
        self.continuous_no_op = 0
        self.predictions = []
        self.thoughts=[]
        self.visited_links = []
        self._page = None

    def _initialize_prompts_pure_vision(self):
        """Specifically for Vision-only agents"""

        return {
            "system_prompt": '''You are assisting humans doing web navigation tasks step by step. At each stage, you can see the webpage by a screenshot and know the previous actions before the current step decided by yourself that have been executed for this task through recorded history. You need to decide on the first following action to take.''',

            "question_description": f'''
    The screenshot below shows the webpage you see. Think step by step before outlining the next action step at the current stage. Clearly outline which element in the webpage users will operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules: 
1. You should only issue a valid action given the current observation. 
2. You should only issue one action at a time
4. Unlike humans, for typing (e.g., in text areas, text boxes), you should try directly typing the input, bypassing the need for an initial click. 
5. You should not attempt to create accounts, log in or do the final submission. 
6. Terminate when you deem the task complete or if it requires potentially harmful actions.
7. Do not generate same action as the previous one, try different ways if keep failing
8. When there is a floating banner like ads, login, or survey floating taking more than 30% of the page, close the floating banner to proceed, the close button could look like a x on the right top corner, or choose NO THANKS to close it.
9. When there is a floating banner on top or bottom of the page like cookie policy taking less than 30% of the page, ignore the banner to proceed.  
10. After typing text into search or text input area, the next action is normally PRESS ENTER
11. When there are bouding boxes in the screenshot, interact with the elements in the bounding boxes
12. When there are multiple clickable buttons having the same value, choose the one with less obstacles in the screenshot.            
                
                
    Here are the descriptions of all allowed actions:

    No Value Operations:
    - CLICK: Click on a webpage element using the mouse.
    - HOVER: Move the mouse over a webpage element without clicking.
    - PRESS ENTER: Press the Enter key, typically to submit a form or confirm an input.
    - SCROLL UP: Scroll the webpage upwards by half of the window height.
    - SCROLL DOWN: Scroll the webpage downwards by half of the window height.
    - PRESS HOME: Scroll to the top of the webpage.
    - PRESS END: Scroll to the bottom of the webpage.
    - PRESS PAGEUP: Scroll up by one window height.
    - PRESS PAGEDOWN: Scroll down by one window height.
    - CLOSE TAB: Close the current tab in the browser.
    - NEW TAB: Open a new tab in the browser.
    - GO BACK: Navigate to the previous page in the browser history.
    - GO FORWARD: Navigate to the next page in the browser history.
    - TERMINATE: End the current task, typically used when the task is considered complete or requires potentially harmful actions.
    - NONE: Indicates that no action is necessary at this stage. Used to skip an action or wait.

    With Value Operations:
    - TYPE: Enter text into a text area or text box. The value is the text to be typed.
    - GOTO: Navigate to a specific URL. The value is the URL to navigate to.
    - SAY: Output answers or other information you want to tell the user.
    - MEMORIZE: Keep some content into action history to memorize it.
    - SELECT: Choose an option from the provided selectors. The value indicates the option to select. If no selector is provided, do not choose this action.
    
    Finally, conclude your answer using the format below. Ensure your answer is strictly adhering to the format provided below. Please do not leave any explanation in your answers of the final standardized format part, and this final part should be clear and certain. The element choice, action, and value should be in three separate lines.
    
    Format:

    ELEMENT: The description about the exact location to help locating the exact position to operate at. Required for click and type.
    
    ACTION: Choose an action from allowed actions.
                
    VALUE: Provide additional input based on ACTION. (If it doesn't involve a value, write "None"            
                
    ''',
    "coordinate_prompt": f"""
Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {{description}}

Answer:""",
    "selector_prompt": f"""
Your task is to help the user select from a list of selectors based on the provided descriptions and actions. If the value to select is not clear, infer the most relevant option based on the context.

Task: {{task}}
History: {{previous}} 
Current Step:
    Selector Descriptions: {{pred_element_label}}
    Value To Select: {{pred_value}}

Avaiable Selectors:
{{options}}

Output Selector id and the most relevant option in that selector in three separate lines.

Selector ID:

Option:
"""
}

    def update_action_space(self, new_actions):
        """Update the action space and regenerate the action_format prompt."""
        if isinstance(new_actions, list) and all(isinstance(item, str) for item in new_actions):
            self.action_space = new_actions
            raise NotImplementedError("Action space update is not yet implemented.")
            self.prompts["action_format"] = f"ACTION: Choose an action from {{{', '.join(self.action_space)}}}."
        else:
            print("Invalid action space provided. It must be a list of strings.")

    def _setup_logger(self, redirect_to_dev_log=False):
        """Set up a logger to log to both file and console within the main_path."""
        logger_name = 'SeeActAgent'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:  # Avoid adding handlers multiple times
            # Create a file handler for writing logs to a file
            log_filename = 'agent.log'
            f_handler = logging.FileHandler(os.path.join(self.main_path, log_filename))
            f_handler.setLevel(logging.INFO)

            # Create a console handler for printing logs to the terminal
            # c_handler = logging.StreamHandler()
            # c_handler.setLevel(logging.INFO)

            # Create formatters for file and console handlers
            file_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_formatter = logging.Formatter('%(message)s')

            # Set formatters for file and console handlers
            f_handler.setFormatter(file_formatter)
            # c_handler.setFormatter(console_formatter)

            # Add the handlers to the logger
            logger.addHandler(f_handler)
            # if not redirect_to_dev_log:  # Only add console handler if not redirecting to dev log
            #     logger.addHandler(c_handler)

        return logger

    async def page_on_close_handler(self):
        # Corrected to use 'self' for accessing class attributes
        if self.session_control['context']:
            try:
                await self.page.title()
            except:
                self.logger.info(
                    "The active tab was closed. Will switch to the last page (or open a new default google page)")
                if self.session_control['context'].pages:
                    self.page = self.session_control['context'].pages[-1]
                    await self.page.bring_to_front()
                    self.logger.info(f"Switched the active tab to: {self.page.url}")
                else:
                    self.page = await self.session_control['context'].new_page()
                    try:
                        await self.page.goto("https://www.google.com/", wait_until="load")
                    except Exception as e:
                        self.logger.info(f"Failed to navigate to Google: {e}")
                    self.logger.info(f"Switched the active tab to: {self.page.url}")

    def save_action_history(self, filename="action_history.txt"):
        """Save the history of taken actions to a file in the main path."""
        history_path = os.path.join(self.main_path, filename)
        with open(history_path, 'w') as f:
            for action in self.taken_actions:
                f.write(action + '\n')
        self.logger.info(f"Action history saved to: {history_path}")

    async def page_on_navigation_handler(self, frame):
        # Corrected to use 'self' for accessing class attributes
        self.page = frame.page

    async def page_on_crash_handler(self, page):
        # Corrected logging method
        self.logger.info(f"Page crashed: {page.url}")
        self.logger.info("Try to reload")
        await page.reload()

    async def page_on_open_handler(self, page):
        # Added 'self' to the handler functions to reference the current instance of the class
        page.on("framenavigated", self.page_on_navigation_handler)
        page.on("close", self.page_on_close_handler)
        page.on("crash", self.page_on_crash_handler)
        self.page: Page = page
        # Additional event listeners can be added here
        try:
            if self.config["agent"]["grounding_strategy"] == "text_choice_som":
                with open(os.path.join(dirname(__file__), "mark_page.js")) as f:
                    mark_page_script = f.read()
                await self.session_control['active_page'].evaluate(mark_page_script)
        except Exception as e:
            pass

    async def start(self, headless=None, args=None, website=None):
        self.playwright = await async_playwright().start()
        self.session_control['browser'] = await normal_launch_async(self.playwright,
                                                                    headless=self.config['browser'][
                                                                        'headless'] if headless is None else headless,
                                                                    args=self.config['browser'][
                                                                        'args'] if args is None else args)
        self.session_control['context'] = await normal_new_context_async(self.session_control['browser'],
                                                                         viewport=self.config['browser'][
                                                                             'viewport'])

        self.session_control['context'].on("page", self.page_on_open_handler)
        await self.session_control['context'].new_page()

        if self.config["basic"]["crawler_mode"] is True:
            await self.session_control['context'].tracing.start(screenshots=True, snapshots=True)

        try:
            await self.page.goto(
                self.config['basic']['default_website'] if website is None else website,
                wait_until="load")
            self.logger.info(f"Loaded website: {self.config['basic']['default_website']}")
        except Exception as e:
            self.logger.info("Failed to fully load the webpage before timeout")
            self.logger.info(e)

            # await asyncio.sleep(2)

    def update_prompt_part(self, part_name, new_text):
        """Update the specified part of the prompt information."""
        raise NotImplementedError("Prompt part update is not yet implemented.")
        if part_name in self.prompts:
            self.prompts[part_name] = new_text
            return True
        else:
            print(f"Prompt part '{part_name}' not found.")
            return False

    def generate_prompt(self, task=None, previous=None, choices=None, do_mtc_grounding=False):

        """Generate a prompt based on the current task, previous actions, and choices."""
        # assert task is not None, "Please input the task."

        prompt_list = []
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage" and not do_mtc_grounding:
            system_prompt_input = self.prompts["system_prompt"]
            question_description_input = self.prompts["question_description"]
            previous_ = self.taken_actions if self.taken_actions else None
            prompt_list.extend(
                generate_new_query_prompt(system_prompt=system_prompt_input,
                                          task=self.tasks[-1], previous_actions=previous_,
                                          question_description=question_description_input))
            return prompt_list
        else:

            system_prompt_input = self.mtc_grounding_prompts["system_prompt"]
            action_space_input = self.mtc_grounding_prompts["action_space"]
            question_description_input = self.mtc_grounding_prompts["question_description"]
            referring_input = self.mtc_grounding_prompts["referring_description"]
            element_format_input = self.mtc_grounding_prompts["element_format"]
            action_format_input = self.mtc_grounding_prompts["action_format"]
            value_format_input = self.mtc_grounding_prompts["value_format"]

            # print(previous)

            previous_ = self.taken_actions if self.taken_actions else None

            # print(previous_)

            prompt_list.extend(
                generate_new_query_prompt(system_prompt=system_prompt_input + "\n" + action_space_input,
                                          task=self.tasks[-1], previous_actions=previous_,
                                          question_description=question_description_input))
            prompt_list.append(
                generate_new_referring_prompt(referring_description=referring_input,
                                              element_format=element_format_input,
                                              action_format=action_format_input, value_format=value_format_input,
                                              choices=choices))

            return prompt_list

    async def perform_action(self, target_element=None, action_name=None, value=None, target_coordinates=None,
                             element_repr=None):

        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage" and action_name != "SELECT":
            selector = "pixel_coordinates"
        elif target_element is not None:
            selector = target_element['selector']
            element_repr = target_element['description']
        else:
            selector = None


        page: Page = self.page

        if action_name == "CLICK" and selector:
            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                # before_content = await self.page.content()
                await self.take_clicking_screenshot(self.page, target_coordinates)
                width, height = self.config['browser']['viewport']['width'], self.config['browser']['viewport']['height']
                x, y = target_coordinates["x"] * width, target_coordinates["y"] * height
                await self.page.mouse.click(round(x), round(y), delay=delay)
                # after_content = await self.page.content()
                # if before_content==after_content:
                #     await target_element['selector'].click(timeout=2000)
                #     self.logger.info(f"Pixel Click Failed, Clicked on element: {element_repr}")
            else:
                await selector.click(timeout=2000)
                self.logger.info(f"Clicked on element: {element_repr}")
        elif action_name == "HOVER" and selector:

            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.take_clicking_screenshot(self.page, target_coordinates)
                width, height = self.config['browser']['viewport']['width'], self.config['browser']['viewport']['height']
                x, y = target_coordinates["x"] * width, target_coordinates["y"] * height
                await self.page.mouse.hover(round(x), round(y), delay=delay)
            else:
                await selector.hover(timeout=2000)
                self.logger.info(f"Hovered over element: {element_repr}")



        elif action_name == "TYPE" and selector:

            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.take_clicking_screenshot(self.page, target_coordinates)
                width, height = self.config['browser']['viewport']['width'], self.config['browser']['viewport']['height']
                x, y = target_coordinates["x"] * width, target_coordinates["y"] * height
                await self.page.mouse.click(round(x), round(y), delay=delay)

                await self.page.keyboard.press("Control+A")
                await self.page.keyboard.press("Backspace")
                # value = stringfy_value(action['fill_text'])
                await self.page.keyboard.type(value)
            else:
                await selector.fill(value)
                await selector.fill(value)
                self.logger.info(f"Typed '{value}' into element: {element_repr}")

        elif action_name == "SCROLL UP":
            await page.evaluate(f"window.scrollBy(0, -{self.config['browser']['viewport']['height'] // 2});")
            self.logger.info("Scrolled up")
        elif action_name == "SCROLL DOWN":
            await page.evaluate(f"window.scrollBy(0, {self.config['browser']['viewport']['height'] // 2});")
            self.logger.info("Scrolled down")
        elif action_name == "PRESS HOME":
            await page.keyboard.press('Home')
            self.logger.info("Pressed Home key")
        elif action_name == "PRESS END":
            await page.keyboard.press('End')
            self.logger.info("Pressed End key")
        elif action_name == "PRESS PAGEUP":
            await page.keyboard.press('PageUp')
            self.logger.info("Pressed PageUp key")
        elif action_name == "PRESS PAGEDOWN":
            await page.keyboard.press('PageDown')
            self.logger.info("Pressed PageDown key")
        elif action_name == "NEW TAB":
            new_page = await self.session_control['context'].new_page()
            # self.session_control['pages'].append(new_page)
            self.logger.info("Opened a new tab")
        elif action_name == "CLOSE TAB":
            await page.close()
            self.logger.info("Closed the current tab")
        elif action_name == "GO BACK":
            await page.go_back()
            self.logger.info("Navigated back")
        elif action_name == "GO FORWARD":
            await page.go_forward()
            self.logger.info("Navigated forward")
        elif action_name == "GOTO" and value:
            await page.goto(value, wait_until="load")
            self.logger.info(f"Navigated to {value}")
        elif action_name == "PRESS ENTER" and selector:
            if selector == "pixel_coordinates":
                delay = random.randint(50, 150)
                await self.take_clicking_screenshot(self.page, target_coordinates)
                width, height = self.config['browser']['viewport']['width'], self.config['browser']['viewport']['height']
                x, y = target_coordinates["x"] * width, target_coordinates["y"] * height
                await self.page.mouse.click(round(x), round(y), delay=delay)
                await page.keyboard.press('Enter')
            else:
                await selector.press('Enter')
                self.logger.info(f"Pressed Enter on element: {element_repr}")
        elif action_name == "PRESS ENTER":
            await page.keyboard.press('Enter')
            self.logger.info(f"Pressed Enter on element: {element_repr}")
        elif action_name == "SELECT" and selector:
            await select_option(selector, value)
            self.logger.info(f"Selected option '{value}' from element: {element_repr}")
        elif action_name == "TERMINATE":
            self.complete_flag = True
            self.exit_reason = "Task completed"
            self.logger.info("Task has been marked as complete. Terminating...")
        elif action_name.upper() in ["NONE"]:
            self.logger.info("No action necessary at this stage. Skipped")
        elif action_name in ["SAY"]:
            self.logger.info(f"Say {value} to the user")
        elif action_name in ["MEMORIZE"]:
            self.logger.info(f"Keep {value} to the action history.")
        else:
            raise Exception(f"Unsupported or improperly specified action: {action_name}")
        if action_name in self.no_element_op and target_element is None:
            new_action = action_name
        else:
            if selector == "pixel_coordinates":
                new_action = element_repr + " -> " + action_name
            else:
                new_action = "[" + target_element['tag_with_role'] + "]" + " "
                new_action += target_element['description'] + " -> " + action_name
        if action_name in self.with_value_op:
            new_action += ": " + value

        # self.dev_logger.info(new_action)
        return new_action
    

    async def gain_elements(self):
        elements = await get_interactive_elements_with_playwright(self.page,
                                                                  self.config['browser']['viewport'])
        
        # '''
        #      0: center_point =(x,y)
        #      1: description
        #      2: tag_with_role: tag_head with role and type # TODO: Consider adding more
        #      3. box
        #      4. selector
        #      5. tag
        #      '''

        elements = sorted(elements, key=lambda el: (
            el["center_point"][1], el["center_point"][0]))  # Sorting by y and then x coordinate

        elements = [{**x, "idx": i, "option": generate_option_name(i)} for i, x in enumerate(elements)]

        return elements
    
    
    async def predict(self):

        """
        Generate a prediction for the next action based on the webpage elements and previous actions.
        """

        self.time_step += 1

        try:
            await self.session_control["active_page"].wait_for_load_state('load')
        except Exception as e:
            pass

        try:
            if self.config["agent"]["grounding_strategy"] == "text_choice_som":
                with open(os.path.join(dirname(__file__), "mark_page.js")) as f:
                    mark_page_script = f.read()
                await self.page.evaluate(mark_page_script)
                await self.page.evaluate("unmarkPage()")
                await self.page.evaluate("""elements => {
                    return window.som.drawBoxes(elements);
                    }""", None)
        except Exception as e:
            self.logger.info(f"Mark page script error {e}")

        # Generate choices for the prompt

        # , self.config['basic']['default_task'], self.taken_actions
        # choices = format_choices(elements)
        # options = format_options(None)

        # print("\n\n",choices)
        # available_selectors = await get_selectors_with_playwright(self.page, viewport_size=self.config['browser']['viewport'])
        prompt = self.generate_prompt(task=self.tasks[-1], previous=self.taken_actions)
        # print("\n\n",prompt)

        # Logging prompt for debugging

        # Capture a screenshot for the current state of the webpage, if required by the model
        screenshot_path = os.path.join(self.main_path, 'trajectories', f'{self.time_step}_full.png')
        self.logger.info(f"Saving screenshot to: {screenshot_path}")
        try:
            await self.page.screenshot(path=screenshot_path, timeout=1000000)
            # await self.take_screenshot(self.page, screenshot_path)
        except Exception as e:
            self.logger.info(f"Failed to take screenshot: {e}")

        terminal_width = 10
        self.logger.info(f"Step - {self.time_step}\n{'-' * terminal_width}\nAction Generation ➡️")
        # for prompt_part in prompt:
        self.logger.info("TASK: " + self.tasks[-1])
        self.logger.info("Previous:")
        for action in self.taken_actions:
            self.logger.info(action)

        output0 = self.engine.generate(prompt=prompt, image_path=self.screenshot_path, turn_number=0)

        terminal_width = 10
        self.logger.info("-" * terminal_width)
        self.logger.info("🤖 Action Generation Output 🤖")

        for line in output0.split('\n'):
            self.logger.info(line)

        terminal_width = 10
        self.logger.info("-" * (terminal_width))
        assert self.config["agent"]["grounding_strategy"] == "pixel_2_stage"

        pred_element_label, pred_action, pred_value = postprocess_action_lmm_pixel(output0)

        if pred_action == "SELECT" or (pred_action == "CLICK" and "dropdown" in pred_element_label.lower()):
            if pred_action == "CLICK":
                self.logger.debug(f"Dropdown click detected, use playwright selector")
            available_selectors = await get_selectors_with_playwright(self.page, viewport_size=self.config['browser']['viewport'])
            format_available_selectors = [
                {"element": f"Selector_{i}", "description": choice['description']}
                for i, choice in enumerate(available_selectors)
            ]
            prompt = self.prompts["selector_prompt"].format(task=self.tasks[-1], pred_element_label=pred_element_label,
                                                            previous=self.taken_actions,
                                                            pred_value=pred_value, options=format_available_selectors)
            response = self.engine.generate(prompt=prompt, text_only=True)

            # parse response
            selector_pattern = re.compile(r"selector_(\d+)", re.IGNORECASE)
            opt_pattern = re.compile(r"Option:(.+)", re.IGNORECASE)
            selector_match = selector_pattern.search(response)
            selector = None
            if selector_match:
                selector_id = int(selector_match.group(1))
                selector = available_selectors[selector_id]
                all_candidates = selector['description'].split("- Options: ")[-1].split(" | ")
            opt = None
            opt_match = opt_pattern.search(response)
            if selector and opt_match:
                opt = opt_match.group(1).strip()
                if opt:
                    opt = difflib.get_close_matches(opt, all_candidates, n=1, cutoff=0.0)[0]
            if selector and opt:
                self.logger.debug(f"Predicted Element: {selector}")
                self.logger.debug(f"Action: SELECT")
                self.logger.debug(f"Value: {opt}")

                prediction = {"action_generation": output0, "action_grounding": response, 
                            "element": selector, "action": "SELECT", "value": opt}
                prediction_json = json.dumps(prediction, indent=4, default=locator_serializer)
                self.predictions.append(prediction_json)
                self.thoughts.append(output0)
                return prediction
            else:
                self.logger.debug(f"Failed to predict SELECT action, move back to pixel grounding")
        # pred_element = get_index_from_option_name(pred_element_label, elements)
        pred_element = None
        # Log the prediction result
        self.logger.debug(f"Retrieved Answer")
        self.logger.debug(f"Predicted Element Description: {pred_element_label}")
        self.logger.debug(f"Action: {pred_action}")
        self.logger.debug(f"Value: {pred_value}")

        # Call a GUI visual grounding model to get the pixel coordinates. For example, UGround, CogAgent, SeeClick.
        text_prompt_for_grouding = self.prompts["coordinate_prompt"].format(description=pred_element_label)
        grounding_response, normalized_coordinates = self.engine.grounding_generate(
            prompt=text_prompt_for_grouding,
            image_path=self.screenshot_path,
        )
        self.logger.info(f"Grounding Response: {grounding_response}\n Coordinates: {normalized_coordinates}")

        prediction = {"action_generation": output0, "action_grounding": output0, "element": pred_element,
                        "action": pred_action, "value": pred_value, 
                        "coordinates": normalized_coordinates,
                        "description": pred_element_label}
            
            

        # else:
        #     options = ""
        #     choice_text = f"Action Grounding ➡️" + "\n" + options
        #     choice_text = choice_text.replace("\n\n", "")

        #     for line in choice_text.split('\n'):
        #         self.logger.info(line)

        #     output = self.engine.generate(prompt=prompt, image_path=self.screenshot_path, turn_number=1,
        #                                   ouput_0=output0)
        #     self.logger.info("🤖 Action Grounding Output 🤖")
        #     for line in output.split('\n'):
        #         self.logger.info(line)

        #     pred_element_label, pred_action, pred_value = postprocess_action_lmm(output)

        #     if len(pred_element_label) in [1, 2]:
        #         element_id = get_index_from_option_name(pred_element_label)
        #     else:
        #         element_id = None
        #     if element_id is not None and element_id < len(elements):
        #         pred_element = elements[element_id]
        #     else:
        #         pred_element = None
        #     # Log the prediction result
        #     self.logger.debug(f"Retrieved Answer")
        #     self.logger.debug(f"Predicted Element: {pred_element}")
        #     self.logger.debug(f"Action: {pred_action}")
        #     self.logger.debug(f"Value: {pred_value}")

        #     prediction = {"action_generation": output0, "action_grounding": output, "element": pred_element,
        #                   "action": pred_action, "value": pred_value}


        # prediction_json = json.dumps(prediction, indent=4)
        self.predictions.append(json.dumps(prediction, indent=4))
        self.thoughts.append(output0)

        # return {"action_generation": output0, "action_grounding": output, "element": pred_element,
        #         "action": pred_action, "value": pred_value}

        return prediction

        # return output0,output,pred_element, pred_action, pred_value

    async def take_clicking_screenshot(self, page, normalized_coordinates):
        client = await page.context.new_cdp_session(page)
        result = await client.send('Page.captureScreenshot', {'format': 'jpeg', 'quality': 100})
        screenshot_base64 = result['data']  
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(io.BytesIO(image_data))
        draw = ImageDraw.Draw(image)

        # Draw a green dot at the click coordinates
        radius = 6
        image_width, image_height = image.size
        x, y = normalized_coordinates["x"] * image_width, normalized_coordinates["y"] * image_height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="green")

        # Save the modified image
        image_path = self.screenshot_path
        click_screenshot_path = image_path.replace('_full.png', '_grounding.png')
        grounding_dir = os.path.join(os.path.dirname(click_screenshot_path), "..", "grounding")
        os.makedirs(grounding_dir, exist_ok=True)
        click_screenshot_path = os.path.join(grounding_dir, os.path.basename(click_screenshot_path))
        image.save(click_screenshot_path)
        self.logger.info(f"Grounding coordinates screenshot to: {click_screenshot_path}")

    async def execute(self, prediction_dict):
        """
        Execute the predicted action on the webpage.
        """

        if prediction_dict is None:
            self.complete_flag = True
            self.exit_reason = "No prediction"
            return
        
        if self.time_step > self.config["agent"]["max_auto_op"]:
            self.complete_flag = True
            self.exit_reason = "Reached step limit"
            self.logger.info(f"the agent reached the step limit {self.config['agent']['max_auto_op']}")
            return
        
        if self.continuous_no_op > self.config["agent"]["max_continuous_no_op"]:
            self.complete_flag = True
            self.exit_reason = "Reached no op limit"
            self.logger.info(f"no executable operations for  {self.config['agent']['max_continuous_no_op']} steps")
            return

        try:
            # Clear the marks before action
            if self.config["agent"]["grounding_strategy"] == "text_choice_som":
                await self.page.evaluate("unmarkPage()")
        except Exception as e:
            pass

        pred_element = prediction_dict["element"]
        pred_action = prediction_dict["action"]
        pred_value = prediction_dict["value"]
        pred_coordinate = None
        pred_element_description=None
        if "description" in prediction_dict:
            pred_element_description=prediction_dict["description"]
        if self.config["agent"]["grounding_strategy"] == "pixel_2_stage" and pred_action != "SELECT":
            pred_coordinate = prediction_dict["coordinates"]

        try:
            if (pred_action not in self.no_element_op) and pred_element == None:
                # self.dev_logger.info
                self.logger.info("DEBUG: WHAT IS PRED ACTION???:" + pred_action)
                # self.dev_logger.info("DEBUG WHAT IS self.no_element_op???:"+ self.no_element_op)
                # pred_action = "NONE"
            new_action = await self.perform_action(pred_element, pred_action, pred_value, pred_coordinate,pred_element_description)
            self.taken_actions.append(new_action)
            if pred_action != "NONE":
                self.valid_op += 1
                self.continuous_no_op = 0
            else:
                self.continuous_no_op += 1
            if self.config["basic"]["crawler_mode"] is True:
                await self.stop_playwright_tracing()
                await self.save_traces()

            return 0
        except Exception as e:

            new_action = f"Failed to perform {pred_action}"

            traceback_info = traceback.format_exc()
            error_message = f"Error executing action {pred_action}: {str(e)}"
            print(traceback_info)
            # exit()
            error_message_with_traceback = f"{error_message}\n\nTraceback:\n{traceback_info}"

            self.logger.info(new_action)
            self.taken_actions.append(new_action)
            self.continuous_no_op += 1
            return 1

    async def stop(self):

        try:
            close_context = self.session_control['context']
            self.session_control['context'] = None
            await close_context.close()
            self.logger.info("Browser context closed.")
        except Exception as e:
            self.logger.info(e)
        final_json = {
            "confirmed_task": self.tasks[0], 
            "website": self.config["basic"]["default_website"],
            "task_id": self.task_id,
            "success_or_not": "" if self.valid_op > 0 else "0",
            "num_step": len(self.taken_actions), 
            "action_history": self.taken_actions,
            "predictions": self.predictions,
            "thoughts": self.thoughts,
            "exit_by": self.exit_reason,
        }

        

        # Using the custom default function in json.dump
        with open(os.path.join(self.main_path, 'all_predictions.json'), 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, default=locator_serializer, indent=4)

        with open(os.path.join(self.main_path, 'result.json'), 'w', encoding='utf-8') as file:
            json.dump(final_json, file, indent=4)
        self.logger.info("Agent stopped.")

        saveconfig(self.config, os.path.join(self.main_path, 'config.toml'))

    def clear_action_history(self):
        """
        Clears the history of actions taken by the agent.
        """
        self.taken_actions.clear()
        self.logger.info("Cleared action history.")

    def reset_comlete_flag(self, flag=False):
        self.complete_flag = flag

    def change_task(self, new_task, clear_history=False):
        """
        Changes the task requirement for the agent.

        Parameters:
        - new_task: The new task requirement as a string.
        """
        if new_task and isinstance(new_task, str):

            self.logger.info(f"Changed task from {self.tasks[-1]} to: {new_task}")
            self.tasks.append(new_task)
            # Optionally clear action history when changing task
            if clear_history:
                self.clear_action_history()
            else:
                self.taken_actions.append(f"Changed task from {self.tasks[-2]} to: {new_task}")

        else:
            self.logger.info("Invalid new task. It must be a non-empty string.")

        # Optionally, you can save the taken_actions to a file or database for record-keeping

    # ADD no op count and op count, add limit to op

    # decompose run to predict and execute.

    # async def take_screenshot(self, page: Page, screenshot_path):    
    #     client = await page.context.new_cdp_session(page)
    #     result = await client.send('Page.captureScreenshot', {'format': 'jpeg', 'quality': 100})
    #     screenshot_base64 = result['data']            
    #     try:
    #         # await self.page.screenshot(path=self.screenshot_path, timeout=1000000)
    #         with open(screenshot_path, 'wb') as f:
    #             f.write(base64.b64decode(screenshot_base64))
    #     except Exception as e:
    #         self.logger.info(f"Failed to take screenshot: {e}")

    async def start_playwright_tracing(self):
        await self.session_control['context'].tracing.start_chunk(
            title=f'Step-{self.time_step}',
            name=f"{self.time_step}"
        )

    async def stop_playwright_tracing(self):
        await self.session_control['context'].tracing.stop_chunk(path=self.trace_path)

    async def save_traces(self):
        # Capture the DOM tree
        dom_tree = await self.page.evaluate("document.documentElement.outerHTML")
        os.makedirs(os.path.join(self.main_path, 'dom'), exist_ok=True)
        with open(self.dom_tree_path, 'w', encoding='utf-8') as f:
            f.write(dom_tree)

        # Capture the Accessibility Tree
        accessibility_tree = await self.page.accessibility.snapshot()
        os.makedirs(os.path.join(self.main_path, 'accessibility'), exist_ok=True)
        with open(self.accessibility_tree_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(accessibility_tree, indent=4))

    @property
    def page(self):
        if self._page is None:
            self._page = self.session_control['active_page']
        return self._page

    @page.setter
    def page(self, value):
        self._page = value

    @property
    def screenshot_path(self):
        return os.path.join(self.main_path, 'trajectories', f'{self.time_step}_full.png')

    @property
    def trace_path(self):
        return os.path.join(self.main_path, 'playwright_traces', f'{self.time_step}.zip')

    @property
    def dom_tree_path(self):
        return os.path.join(self.main_path, 'dom', f'{self.time_step}.html')

    @property
    def accessibility_tree_path(self):
        return os.path.join(self.main_path, 'accessibility', f'{self.time_step}.json')
