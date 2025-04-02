import os
os.environ["OPENAI_API_KEY"] = os.environ["WEB_LDBD_API_KEY"]

import asyncio
from seeact.agent import SeeActAgent
import json
from tqdm import tqdm
# Setup your API Key here, or pass through environment

# TASK_FILE_PATH = "data/filtered_test_data_300.json"
TASK_FILE_PATH = "data/FINAL_selected_tasks_300.json"
SAVE_DIR = f"output/0223"
OLD_SAVE_DIR = f"auto_eval/results/uground"

default_seeact_kwargs = {
    "grounding_strategy": "pixel_2_stage",
    "model": "gpt-4o",
    "temperature": 0.0,
    "max_auto_op": 20,
    "max_continuous_no_op": 2,
    "headless": False,
    "grounding_model_config": {
        "model": "osunlp/UGround-V1-7B",
        "base_url": "http://a0023:6999/v1",
        # "base_url": "http://localhost:6999/v1",
        "api_key": "skdummy",
    }
}
import logging
import glob
async def run_agent(single_query_task, force_restart=False):


    # for single_query_task in tqdm(query_tasks[start_index:end_index]):
    confirmed_task = single_query_task["confirmed_task"]
    confirmed_website = single_query_task["website"]
    task_id = single_query_task["task_id"]
    print(f"Task ID: {task_id}")
    save_file_dir = os.path.join(SAVE_DIR, task_id)

    # glob every subfolder in the save_file_dir
    old_save_file_dir = os.path.join(OLD_SAVE_DIR, task_id)
    if os.path.exists(old_save_file_dir) and not force_restart:
        result_path = os.path.join(old_save_file_dir, "result.json")
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as file:
                result = json.load(file)
                task_in_result = result["confirmed_task"]
                if task_in_result == confirmed_task:
                    print(f"Task {task_id} already completed.")
                    return
                else:
                    print(f"Task {task_id} already exists but with different task {task_in_result}.")
        else:
            print(f"Task {task_id} folder already exists but without result.")
    else:
        print(f"Task {task_id} not exists.")

    # os.makedirs(main_result_path, exist_ok=True)
    seeact_kwargs = default_seeact_kwargs.copy()
    agent = SeeActAgent(
        save_file_dir=save_file_dir,
        default_task=confirmed_task,
        default_website=confirmed_website,
        **seeact_kwargs,
    )
    setattr(agent, "task_id", task_id)
    
    agent.logger.setLevel(logging.INFO)
    for log_handler in agent.logger.handlers:
        log_handler.setLevel(logging.INFO)
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
    parser.add_argument("--index", type=int, default=None, help="Index of task to process.")
    parser.add_argument("--task_id", type=str, default=None, help="Task ID to process.")
    parser.add_argument("--force_restart", action="store_true", help="Force restart the agent.")
    parser.add_argument("--do_debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()
    with open(TASK_FILE_PATH, 'r', encoding='utf-8') as file:
        query_tasks = json.load(file)
    if args.do_debug:
        logging.basicConfig(level=logging.DEBUG)
        import debugpy
        debugpy.listen(("0.0.0.0", 5679))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    if args.index is not None:
        asyncio.run(run_agent(query_tasks[args.index], force_restart=args.force_restart))
    elif args.task_id is not None:
        for task in query_tasks:
            if task["task_id"] == args.task_id:
                asyncio.run(run_agent(task, force_restart=args.force_restart))
                break
    # asyncio.run(run_agent(query_tasks[args.index]))