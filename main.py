import asyncio
import os
from seeact.agent import SeeActAgent
import json
from tqdm import tqdm
# Setup your API Key here, or pass through environment
os.environ["OPENAI_API_KEY"] = os.environ["WEB_LDBD_API_KEY"]

# TASK_FILE_PATH = "data/filtered_test_data_300.json"
TASK_FILE_PATH = "data/selected_tasks_150.json"
SAVE_DIR = f"output/selected_tasks_150"

default_seeact_kwargs = {
    "grounding_strategy": "pixel_2_stage",
    "model": "gpt-4o",
    "temperature": 0.0,
    "max_auto_op": 30,
    "max_continuous_no_op": 2,
    "headless": False,
    "grounding_model_config": {
        "model": "osunlp/UGround-V1-7B",
        "base_url": "http://localhost:6999/v1",
        "api_key": "skdummy",
    }
}
import logging

async def run_agent(start_index=0, end_index=-1):
    
    with open(TASK_FILE_PATH, 'r', encoding='utf-8') as file:
        query_tasks = json.load(file)

    for single_query_task in tqdm(query_tasks[start_index:end_index]):
        confirmed_task = single_query_task["confirmed_task"]
        confirmed_website = single_query_task["website"]
        task_id = single_query_task["task_id"]
        save_file_dir = os.path.join(SAVE_DIR, task_id)
        # os.makedirs(main_result_path, exist_ok=True)
        seeact_kwargs = default_seeact_kwargs.copy()
        agent = SeeActAgent(
            save_file_dir=save_file_dir,
            default_task=confirmed_task,
            default_website=confirmed_website,
            **seeact_kwargs,
        )
        setattr(agent, "task_id", task_id)
        
        agent.logger.setLevel(logging.DEBUG)
        for log_handler in agent.logger.handlers:
            log_handler.setLevel(logging.DEBUG)
        agent.logger.info(f"Model: {agent.engine.model_names}")
        await agent.start()
        while not agent.complete_flag:
            prediction_dict = await agent.predict()
            await agent.execute(prediction_dict)        
        await agent.stop()
        agent.logger.info(f"Task {task_id} completed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SeeActAgent with task range control.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of tasks to process (inclusive).")
    parser.add_argument("--end_index", type=int, default=None, help="End index of tasks to process (exclusive).")
    
    args = parser.parse_args()
    asyncio.run(run_agent(args.start_index, args.end_index))