import os
import sys
import traceback
from pathlib import Path
import yaml

from datetime import datetime

from langchain.chat_models import init_chat_model

from langchain_core.messages import HumanMessage
from mcp.server.fastmcp import FastMCP
import hpc_runner as runner
from typing import Optional
from tavily import TavilyClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

os.environ['LSF_SERVERDIR'] = '/usr/local/lsf/10.1/linux3.10-glibc2.17-x86_64/etc'
os.environ['LSF_LIBDIR'] = '/usr/local/lsf/10.1/linux3.10-glibc2.17-x86_64/lib'
os.environ['LSF_BINDIR'] = '/usr/local/lsf/10.1/linux3.10-glibc2.17-x86_64/bin'
os.environ['LSF_ENVDIR'] = '/usr/local/lsf/conf'

lsf_bin = os.environ['LSF_BINDIR']
if lsf_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = f"{lsf_bin}:{os.environ.get('PATH', '')}"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
hpc_mcp = FastMCP("HPC_Job_Manager")
class HPCApplicationManager:
    """
        Manages  HPC applications path and add configurations from .env
        """
    APP_PREFIXES = {
        'vasp': 'VASP',
        'lammps': 'LAMMPS',
    }
    APP_KEYWORDS = {
        'vasp': ['vasp', 'VASP', 'vasp_std', 'vasp_gam', 'vasp_ncl'],
        'lammps': ['lammps', 'LAMMPS', 'lmp', 'lmp_cpu', 'lmp_gpu', 'in.', '.lmp'],
    }
    def __init__(self):
        self.app_configs = self._load_all_app_configs()
    def _load_all_app_configs(self):
        configs ={}
        for app_name,prefix in self.APP_PREFIXES.items():
            app_config = self._load_app_config(prefix)
            if app_config:
                configs[app_name] = app_config
        return configs
    def _load_app_config(self,prefix:str):
        config ={}
        version_key = f"{prefix}_VERSION"
        config['version_key'] = version_key
        exec_key = f"{prefix}_EXECUTABLE"
        executable = os.environ.get(exec_key)

        path_key = f"{prefix}_PATH"
        path = os.getenv(path_key)
        if not path:
            print(f"[WARNING] {prefix}_PATH environment variable not set",file=sys.stderr)
            return None
        if not executable:
            print(f"[WARNING] {exec_key} not set in environment", file=sys.stderr)
            return None

        config={
            'path': path,
            'executable': executable,
            'modules':self._get_modules(prefix),
        }
        return config
    def _get_modules(self,prefix):
        modules_key = f"{prefix}_MODULES"
        modules_str = os.getenv(modules_key, '')
        return modules_str.split(',') if modules_str else []

    def get_app_command(self,app_name:str, input_file:Optional[str]=None):
        if app_name not in self.app_configs:
            raise ValueError(f"Application {app_name} not configured in environment")
        app_config = self.app_configs[app_name]
        app_path = app_config['path']
        executable = app_config['executable']
        if app_name =='vasp':
            command = f"mpirun {executable}"
            return {
                'command': command,
                'modules': app_config['modules'],
                'path': app_path,
                'executable': executable,
                'export_path': f"{app_path}/bin:$PATH"
            }

        elif app_name == 'lammps':
            if not input_file:
                raise ValueError("ERROR: Missing input file for LAMMPS")

            command = f"mpirun {app_path}/src/{executable} < {input_file}"
            return {
                'command': command,
                'modules': app_config['modules'],
                'path': app_path,
                'executable': executable,
            }
        else:
            raise ValueError(f"Unsupported application: {app_name}")





app_manager = HPCApplicationManager()

@hpc_mcp.tool()
def submit_and_monitor(config_path:str) -> str:
    """
    Submits and monitors a job based on a YAML configuration file.
    This is the primary tool for starting a job run.
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return f"ERROR: Configuration file {config_path} does not exist"
        cfg = yaml.safe_load(config_file.read_text())
        sites = cfg.get("sites", {})
        jobs = cfg.get("jobs", {})
        scheduler = runner.detect_scheduler(sites)
        print(f"[LOG] scheduler is {scheduler}\n", file=sys.stderr, flush=True)
        spec = runner.merge_spec(sites.get("defaults", {}), jobs)
        jobid = runner.submit(spec, scheduler)
        print(f"[LOG] Job {jobid} submitted. Starting to monitor...",file=sys.stderr,flush=True)
        rr  = runner.monitor(scheduler, spec,jobid,poll_s=30)
        result_message = (
            f"JOB_RESULT: {'SUCCESS' if rr.final_state.value == runner.JobState.COMPLETED else 'FAILED'}"
            f"\nJOB_ID: {rr.jobid}"
            f"\nFINAL_STATE: {rr.final_state.value}"
            f"\nOUT_FILE: {rr.out_file}"
            f"\nERR_FILE: {rr.err_file}"
        )
        print(f"FINAL JOB STATUS: {rr.final_state.value}", file=sys.stderr, flush=True)
        if rr.final_state.value == runner.JobState.COMPLETED:
            print(f"Success!", file=sys.stderr, flush=True)
            if rr.out_file and rr.out_file.exists():
                output_preview = rr.out_file.read_text(encoding='utf-8',error='ignore')[:1000]
                result_message += f"\nOUTPUT_PREVIEW:\n {output_preview}"
                if rr.out_file.stat().st_size >1000:
                    result_message += f"\n... (output truncated, use 'read_job_output' for full output)"
                else:
                    result_message +="\n\n(No output file found)"

        elif rr.final_state.value == runner.JobState.FAILED:
            print(f"FAILED!", file=sys.stderr, flush=True)
            if rr.err_file and rr.err_file.exists():
                error_preview = rr.err_file.read_text(encoding='utf-8', errors='ignore')[:500]
                result_message += f"\nERROR_PREVIEW: {error_preview}..."
                if rr.err_file.stat().st_size > 1000:
                    result_message += f"\n... (error truncated, use 'read_job_output' for full error log)"
            result_message += "\n\nNEXT_STEPS: Use 'read_config' to examine the configuration, then 'update_config' to fix issues, and 'submit_and_monitor_job' to retry."
        else:
            print("Unknown final state", file=sys.stderr, flush=True)
        return result_message
    except Exception as e:
        return f"CRITICAL_ERROR: Job submission failed before monitoring could start. Details: {e}"

@hpc_mcp.tool()
def read_config(config_path:str) -> str:
    '''
    Read and return the contents of a job configuration file.
    This allows the LLM to analyze the current configuration.
    '''
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return f"ERROR: Configuration file {config_path} does not exist"
        content = config_file.read_text()
        return f"CONFIG_CONTENT:\n{content}"
    except Exception as e:
        return f"ERROR: Unable to read configuration file: {e}"

@hpc_mcp.tool()
def update_config(config_path:str, updated_yaml:str) -> str:
    """
    Update a job configuration file with corrected settings.
    The LLM can use this to fix issues identified from error messages.

    Args:
        config_path (str): Path to the job configuration file.
        updated_yaml (str): The updated job configuration file.
    """
    try:
        config_file = Path(config_path)
        try:
            yaml.safe_load(updated_yaml)
        except yaml.YAMLError as e:
            return f"ERROR: The provided text is not valid YAML: {e}"
        backup_path = config_file.with_suffix(
            config_file.suffix + f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        if config_file.exists():
            backup_path.write_text(config_file.read_text())
            print(f"[LOG] Created backup at {backup_path}", file=sys.stderr, flush=True)
        config_file.write_text(updated_yaml)
        return f"SUCCESS: Updated job configuration file {config_file}\nBacked up at {backup_path}\nYou can now resubmit the job using 'submit_and_monitor_job'"
    except Exception as e:
        return f"ERROR: Failed to update job configuration file: {e}"

@hpc_mcp.tool()
def read_job_output(file_path:str, max_lines:Optional[int]=1000) -> str:
    """
    Read the output or error file from a completed job.
    Useful for detailed error analysis
    Args:
        file_path (str): Path to the job output/error file.
        max_lines (Optional[int]): Maximum number of lines to read.
    """
    try:
        file = Path(file_path)
        if not file.exists():
            return f"ERROR: Job output file {file_path} does not exist"
        lines = file.read_text(encoding='utf-8',errors='ignore').splitlines()
        if max_lines and len(lines) >= max_lines:
            content = '\n'.join(lines[:max_lines])
            return f"FILE_CONTENT (first {max_lines} lines):\n{content}\n\n... ({len(lines) - max_lines} more lines truncated)"
        else:
            return f"FILE_CONTENT:\n" + '\n'.join(lines)
    except Exception as e:
        return f"ERROR: Failed to read job output file: {e}"

@hpc_mcp.tool()
async def search_policy(query:str)->str:
    """
    Searches NCSU HPC documentation to find specific information like module names,
    run commands, and queue policies. It first uses a web search to find the
    relevant documentation page, then directly scrapes that page for code blocks
    and specific details to get the most accurate information.
    """
    try:
        api_key = os.environ['TAVILY_API_KEY']
        if not api_key:
            return f"ERROR: TAVILY_API_KEY environment variable not set"
        try:
            tavily = TavilyClient(api_key=api_key)

        except Exception as e:
            return f"ERROR: Failed to initialize Tavily client: {e}"

        try:
            model = init_chat_model(model="openai:gpt-4.1-mini")
        except Exception as e:
            return f"ERROR: Failed to initialize LLM model: {e}"

        print(f"[LOG] Searching for: {query}", file=sys.stderr, flush=True)
        try:
            search_results = tavily.search(f"site:hpc.ncsu.edu {query}", max_results=3, include_raw_content=True)
        except Exception as e:
            return f"ERROR: Tavily search failed: {e}"

        good_results = [
            r for r in search_results['results']
            if r.get("score",0) > 0.5
        ]

        if not good_results:
            return (
                f"ERROR: No high-quality results found for query '{query}'. "
                "Try rephrasing your search or check if the topic exists in NCSU HPC documentation."
            )
        query_embedding = embedding_model.encode(query)
        all_chunks = []
        for result in good_results:
            content = result.get("raw_content") or result.get("content","")
            if not content.strip():
                continue
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            chunk_embedding = embedding_model.encode(chunks)
            consine_score = util.cos_sim(query_embedding, chunk_embedding)[0]
            tavily_score = result.get("score",0)
            for i, chunk in enumerate(chunks):
                embedding_score = consine_score[i].item()
                combined_score = (0.3 * tavily_score) + (0.7 * embedding_score)

                all_chunks.append({
                    "url": result["url"],
                    "title":result["title"],
                    "chunk":chunk,
                    "tavily_score":tavily_score,
                    "embedding_score":embedding_score,
                    "combined_score":combined_score,
                })
        all_chunks.sort(key=lambda x: x["combined_score"], reverse=True)
        top_chunks = all_chunks[:5]

        print(f"[LOG] Top chunk scores: {[c['combined_score'] for c in top_chunks]}",file=sys.stderr)

        context = "\n\n---\n\n".join([
            f"SOURCE: {c['title']} (relevance: {c['combined_score']:.2f})\n"
            f"URL: {c['url']}\n"
            f"CONTENT:\n{c['chunk']}"
            for c in top_chunks
        ])
        summary_prompt = (
            f"You are an expert HPC assistant. Based on the following highly relevant sections from the HPC documentation, "
            f"provide a final, concise answer to the user's original query.\n\n"
            f"QUERY: \"{query}\"\n\n"
            f"RELEVANT SECTIONS:\n---\n{context}\n---\n\n"
            "Provide exact resource limits:"
        )
        try:
            summary = await model.ainvoke([HumanMessage(content=summary_prompt)])
        except Exception as e:
            return f"ERROR: LLM failed to generate summary: {e}"
        unique_urls = list(set(c['url'] for c in top_chunks))
        sources = "\n".join([f"  - {url}" for url in unique_urls])

        return f"{summary.content.strip()}\n\nSOURCES:\n{sources}"
    except Exception as e:
        import trackback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Unexpected error in search_policy: {error_trace}", file=sys.stderr, flush=True)
        return f"ERROR: Failed to search HPC policy. Details: {e}"
@hpc_mcp.tool()
def check_config_exists()->str:
    """
    Check if a config.yaml file exists in the give directory.
    """
    try:
        config_path = Path("config.yaml")
        if config_path.exists():
            return f"SUCESS: config.yaml already exists"
        else:
            return f"INFO: config.yaml does not exist. You should create one"
    except Exception as e:
        return f"ERROR: Failed to check config.yaml: {e}"
@hpc_mcp.tool()
def create_job_config(
        config_path: str,
        work_dir: str,
        job_params: dict,
        scheduler_type: str,
        engine: str,
        job_name: str,
        input_file: Optional[str] = None,
):
    """
        Creates a new job config.yaml file from a dictionary of parameters.
        This tool is flexible and can accept any valid job parameter.
        For memory, specify 'mem' (memory in GB).
        Example job_params:
        {
            "job_name": "dft_relax",
            "ncores": 16,
            "time": "48:00:00",
            "mem": 32,
            "modules": ["vasp/6.4.1"],
            "account": "my_project_account_123",
            "qos": "high_priority"
            "env": {Path:path}

        }

        Args:
            config_path (str): The path to the YAML file to create, e.g., 'config.yaml'.
            work_dir (str): The working directory where the job will run.
            job_params (dict): A dictionary containing all job specifications.
    """
    try:
        if not isinstance(job_params, dict):
            return "ERROR: job_params must be a dictionary"
        if not work_dir:
            return "ERROR: work_dir is required"
        if not job_name:
            return "ERROR: job_name is required"
        if not engine:
            return "ERROR: engine is required"

        if engine == 'lammps':
            if not input_file:
                return "ERROR: LAMMPS require an input file"
            config_update = app_manager.get_app_command(app_name="lammps", input_file=input_file)
        else:
            config_update = app_manager.get_app_command(app_name="vasp")
            job_params["env"] = {'PATH': config_update["export_path"]}


        job_params["modules"] = config_update["modules"]


        config_data = {
            "sites": {
                "type": scheduler_type,
                "defaults": dict(job_params),
            },
            "jobs": {
                "work_dir": work_dir,
                "job_name": job_name,
                "command": config_update["command"],
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, sort_keys=False)
        return f"SUCCESS: Created job configuration file {config_path}"
    except Exception as e:
        return f"ERROR: Failed to create job configuration file: {e}"


if __name__ == "__main__":
    hpc_mcp.run()
