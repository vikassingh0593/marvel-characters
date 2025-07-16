
import argparse
import os
import requests
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument("--org", type=str, required=True, help="GitHub organization")
parser.add_argument("--repo", type=str, required=True, help="GitHub repository")
parser.add_argument("--git_sha", type=str, required=True, help="git sha of the commit")
parser.add_argument("--job_run_id", type=str, required=True, help="run id of the run of the databricks job")
parser.add_argument("--job_id", type=str, required=True, help="id of the job")

args = parser.parse_args()

host = get_dbr_host()

org = args.org
repo = args.repo
git_sha = args.git_sha
job_id = args.job_id
run_id = args.job_run_id

token = os.environ["TOKEN_STATUS_CHECK"]
url = f"https://api.github.com/repos/{org}/{repo}/statuses/{git_sha}"
link_to_databricks_run = f"{host}/jobs/{job_id}/runs/{run_id}"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28"
}

payload = {
    "state": "success",
    "target_url": f"{link_to_databricks_run}",
    "description": "Marvel integration test is successful!", 
    "context": "integration-testing/marvel-databricks"
}

response = requests.post(url, headers=headers, json=payload)
logger.info("Status:", response.status_code)
