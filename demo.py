import asyncio
import os
from seeact.agent import SeeActAgent
import json
# Setup your API Key here, or pass through environment
os.environ["OPENAI_API_KEY"] = os.environ["WEB_LDBD_API_KEY"]

# TASK_FILE_PATH = "data/online_tasks/sample_tasks.json"
TASK_FILE_PATH = "data/filtered_test_data_300.json"
# CONFIG_PATH = "config/online300.toml"
SAVE_DIR = "output/sample_online"

SAVE_FILE_SUFFIX = "4o_ug_7b"

default_seeact_kwargs = {
    "grounding_strategy": "pixel_2_stage",
    "model": "gpt-4o",
    "temperature": 0.0,
    "headless": True,
    "max_auto_op": 30,
    "max_continuous_no_op": 2,
    "grounding_model_config": {
        "model": "osunlp/UGround-V1-7B",
        "base_url": "http://localhost:6999/v1",
        "api_key": "skdummy",
    }
}
import logging
async def run_agent():
    
    with open(TASK_FILE_PATH, 'r', encoding='utf-8') as file:
        query_tasks = json.load(file)

    for i, single_query_task in enumerate(query_tasks):
        confirmed_task = single_query_task["confirmed_task"]
        confirmed_website = single_query_task["website"]
        task_id = single_query_task.get("task_id", f"sample_task_{i}")
        save_file_dir = os.path.join(SAVE_DIR, f"{task_id}_{SAVE_FILE_SUFFIX}")
        # os.makedirs(main_result_path, exist_ok=True)

        agent = SeeActAgent(
            save_file_dir=save_file_dir,
            default_task=confirmed_task,
            default_website=confirmed_website,
            **default_seeact_kwargs,
        )
        agent.logger.setLevel(logging.DEBUG)
        for log_handler in agent.logger.handlers:
            log_handler.setLevel(logging.DEBUG)
        agent.logger.debug(f"Start running task: {task_id}, task: {confirmed_task}, website: {confirmed_website}")
        await agent.start()
        while not agent.complete_flag:
            prediction_dict = await agent.predict()
            await agent.execute(prediction_dict)
        await agent.stop()
        break
    

if __name__ == "__main__":
    asyncio.run(run_agent())