from pathlib import Path
import yaml
from enum import Enum
from typing import Dict, Any
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List,Optional, Tuple
import os
import subprocess
import argparse
import time
import re
import sys

@dataclass
class JobSpec:
    work_dir:str
    job_name:str
    command:str
    ncores: int=32
    time: str = "01:00:00"
    mem: Optional[int] = None
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    queue: Optional[str] = None
    gpus: Optional[int] = None
    modules:List[str] = field(default_factory=list)
    span_one_host: bool = False
    env: Dict[str, Any] = field(default_factory=dict)
    comments: List[str] = field(default_factory=list)




class JobState(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN   = "UNKNOWN"


@dataclass
class RunResult:
    jobid: str
    final_state: JobState
    out_file: Optional[Path] = None
    err_file: Optional[Path] = None


class ScheduleKind(str,Enum):
    SLURM = "slurm"
    LSF = "lsf"

def build_submit_args(scheduler: ScheduleKind, script: Path) -> List[str]:
    if scheduler == ScheduleKind.SLURM:
        return ["sbatch", str(script)]
    elif scheduler == ScheduleKind.LSF:
        # run bsub through a shell so "<" redirection works
        return ["bash", "-lc", f"bsub < {script}"]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

def which(cmd:str)->bool:
    return shutil.which(cmd) is not None
def detect_scheduler(site_cfg:Dict[str,Any])-> ScheduleKind:
    type = (site_cfg.get("type") or "")
    if type in ("slurm", "lsf"):
        return ScheduleKind.SLURM if type == "slurm" else ScheduleKind.LSF
    if which("sbatch"):
        return ScheduleKind.SLURM
    if which("bsub"):
        return ScheduleKind.LSF
    raise ValueError("Unknown scheduler type")


def run_cmd(args) -> Tuple[int, str, str]:
    p = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def merge_spec(site_defaults:Dict[str,Any],job_cfg:Dict[str,Any] ) -> JobSpec:
    data = {**site_defaults,**job_cfg}
    modules = data.get("modules") or []
    env = data.get("env") or {}
    comments = data.get("comments") or []
    return JobSpec(
        work_dir=data["work_dir"],
        job_name=data["job_name"],
        command=data["command"],
        ncores=int(data.get("ncores", 32)),
        time=str(data.get("time", "02:00:00")),
        mem=(int(data["mem"]) if data.get("mem") not in (None, "", 0) else None),
        partition=data.get("partition"),
        account=data.get("account"),
        qos=data.get("qos"),
        queue=data.get("queue"),
        gpus=(int(data["gpus"]) if data.get("gpus") not in (None, "", 0) else None),
        modules=list(modules),
        span_one_host=bool(data.get("span_one_host", False)),
        env=dict(env),
        comments=list(comments),
    )


def _lsf_time_from_slurm(t:str) -> str:
    parts = t.split(":")
    if len(parts) >= 2:
        return f"{parts[0]}:{parts[1]}"
    return t

def _slurm_script(spec:JobSpec) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {spec.job_name}",
        f"#SBATCH -o {spec.job_name}.%j.out",
        f"#SBATCH -e {spec.job_name}.%j.err",
        f"#SBATCH -n {spec.ncores}",
        f"#SBATCH -t {spec.time}",
    ]
    if spec.partition:  lines.append(f"#SBATCH -p {spec.partition}")
    if spec.account:    lines.append(f"#SBATCH -A {spec.account}")
    if spec.qos:        lines.append(f"#SBATCH --qos={spec.qos}")
    if spec.mem:     lines.append(f"#SBATCH --mem={spec.mem}")
    if spec.gpus:       lines.append(f"#SBATCH --gres=gpu:{spec.gpus}")

    body = []
    body.append("set -euo pipefail")
    for k, v in spec.env.items():
        body.append(f'export {k}="{v}"')
    if spec.modules:
        body.append("module purge || true")
        body.extend([f"module load {m}" for m in spec.modules])
    body += [f"cd {spec.work_dir}", spec.command]
    #body +=[spec.command]
    return "\n".join(lines + [""] + body) + "\n"

def _lsf_script(spec:JobSpec) -> str:
    lines = [
        "#!/bin/bash",
        f"#BSUB -J {spec.job_name}",
        f"#BSUB -o {spec.job_name}.%J.out",
        f"#BSUB -e {spec.job_name}.%J.err",
        f"#BSUB -n {spec.ncores}",
        f"#BSUB -W {_lsf_time_from_slurm(spec.time)}",
    ]
    if spec.comments:
        lines.append("")
        for comment in spec.comments:
            lines.append(f"#{comment}")
    if spec.queue:
        lines.append(f"#BSUB -q {spec.queue}")
    if spec.span_one_host:
        lines.append('#BSUB -R "span[hosts=1]"')
    if spec.mem:
        lines.append(f'#BSUB -R "rusage[mem={spec.mem}]"')
    body = []
    body.append("set -euo pipefail")
    for k,v in spec.env.items():
        body.append(f'export {k}="{v}"')
    if spec.modules:
        body.append("module purge || true")
        body.extend([f"module load {m}" for m in spec.modules])
    body +=[f"cd {spec.work_dir}", spec.command]
    #body += [spec.command]
    return "\n".join(lines + [""] + body) + "\n"

def parse_lsf_jobid(out:str, err:str) -> str|None:
    #m = re.search(r"<(\d+)>", (out or "") + " " + (err or ""))
    print(out)
    m = re.search(r"Job <(\d+)> is submitted", out + err)
    return m.group(1) if m else None

def parse_slurm_jobid(out: str) -> str | None:
    m = re.search(r"Submitted batch job (\d+)", out)
    return m.group(1) if m else None

def write_script(scheduler:ScheduleKind, spec: JobSpec) -> Path:
    Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
    script= Path(spec.work_dir)/f'submit_{spec.job_name}.sh'
    if scheduler == ScheduleKind.SLURM:
        text = _slurm_script(spec)
    else:
        text = _lsf_script(spec)
    script.write_text(text)
    os.chmod(script, 0o755)
    return script

def submit_job(scheduler:ScheduleKind, script: Path) -> str:
    args = build_submit_args(scheduler, script)
    code, out, err = run_cmd(args)
    #jobid = parse_lsf_jobid(out, err)
    if code != 0:
        raise RuntimeError(f"Job submission failed:\n{out}\n{err}")
    #print(f"[Important] the out output is: {out}\n")
    if scheduler == ScheduleKind.SLURM:
        jobid = parse_slurm_jobid(out)
    else:
        jobid = parse_lsf_jobid(out, err)
    if not jobid:
        raise RuntimeError(f"Could not parse job ID from submission output:\nSTDOUT: {out}\nSTDERR: {err}")

    print(f"[Important] the out output is: {out}\n")
    return jobid

def slurm_status(jobid: str) -> JobState:
    # Try squeue first
    code, out, _ = run_cmd(["squeue", "-j", jobid, "-h", "-o", "%T"])
    if code == 0 and out:
        state = out.strip().upper()
        if state in ("PENDING", "CONFIGURING"): return JobState.PENDING
        if state in ("RUNNING", "COMPLETING"):  return JobState.RUNNING
        if state == "COMPLETED":                return JobState.COMPLETED
        if state in ("FAILED","CANCELLED","TIMEOUT","NODE_FAIL","PREEMPTED","OUT_OF_MEMORY"):
            return JobState.FAILED
        return JobState.UNKNOWN
    # If not in squeue, try sacct for terminal state
    if which("sacct"):
        code, out, _ = run_cmd(["sacct", "-j", jobid, "--format=State,ExitCode", "-P", "-n"])
        if code == 0 and out:
            first = out.splitlines()[0]
            state = first.split("|", 1)[0].upper()
            if state.startswith("COMPLETED"): return JobState.COMPLETED
            if any(state.startswith(x) for x in ("FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED")):
                return JobState.FAILED
            if "COMPLETED" in state: return JobState.COMPLETED
    return JobState.UNKNOWN

def lsf_status(jobid:str) -> JobState:
    code, out, err = run_cmd(["bjobs", "-noheader", "-o", "stat", jobid])
    print(f"[LOG] code: {code}, out: {out}, err: {err}",file=sys.stderr,flush=True)
    '''if code != 0:
        if "Job <{}> is not found".format(jobid) in err or "not found" in err.lower():
            return JobState.UNKNOWN
        return JobState.UNKNOWN'''
    if "is not found" in err:
        # It's not running, check accounting for final status
        code, out, _ = run_cmd(["bacct", "-l", jobid])
        if "Done successfully" in out:
            return JobState.COMPLETED
        if "Exited with exit code" in out:
            return JobState.FAILED
        return JobState.UNKNOWN  # Truly unknown state

    state = out.strip().split()[-1].upper()
    print(f"state: {state}",file=sys.stderr,flush=True)
    if state in ("PEND", "PSUSP", "USUSP", "SSUSP"):
        return JobState.PENDING
    if state in ("RUN", "UNKWN"):
        return JobState.RUNNING
    if state == "DONE":
        return JobState.COMPLETED
    if state in ("EXIT", "ZOMBI"):
        return JobState.FAILED
    return JobState.UNKNOWN


def job_status(scheduler: ScheduleKind, jobid: str) -> JobState:
    if scheduler == ScheduleKind.SLURM:
        print(f"[slurm] checking {jobid}\n",file=sys.stderr, flush=True)
        return slurm_status(jobid)
    else:
        print(f"[lsf] checking {jobid}\n",file=sys.stderr, flush=True)
        return lsf_status(jobid)

def cancel_job(scheduler: ScheduleKind, jobid: str) -> None:
    run_cmd(["scancel", jobid]) if scheduler == ScheduleKind.SLURM else run_cmd(["bkill", jobid])

def submit(spec: JobSpec, scheduler: ScheduleKind) -> str:
    script = write_script(scheduler, spec)
    jobid = submit_job(scheduler, script)   # uses sbatch or bsub; returns immediately
    Path(spec.work_dir, "JOBID").write_text(jobid + "\n")
    print(f"Submitted {jobid}. Script: {script}", file=sys.stderr, flush=True)
    return jobid

def monitor(scheduler: ScheduleKind, spec: JobSpec, jobid: str, poll_s: int) -> RunResult:
    """Monitors a job until it reaches a terminal state."""
    out_file = Path(spec.work_dir) / f"{spec.job_name}.{jobid}.out"
    err_file = Path(spec.work_dir) / f"{spec.job_name}.{jobid}.err"
    last = None
    while True:
        state = job_status(scheduler, jobid)
        if state != last:
            print(f"Job {jobid} state: {state}", file=sys.stderr, flush=True)
            last = state
        if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            return RunResult(jobid, state, out_file, err_file)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--monitor", help="Monitor an existing job by jobid")
    ap.add_argument("--write-only", action="store_true",
                    help="Only write the submission script and exit (no submit)")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--cancel")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sites = cfg["sites"] or {}
    jobs = cfg["jobs"] or {}

    scheduler = detect_scheduler(sites)
    spec = merge_spec(sites.get("defaults", {}), jobs)

    if args.cancel:
        cancel_job(scheduler, args.cancel)
        print(f"Cancelled {args.cancel}")
        return

    if args.monitor:
        rr = monitor(scheduler, spec, args.monitor, poll_s=args.poll)
        print(f"Job {rr.jobid} finished: {rr.final_state}")
        if rr.out_file and rr.out_file.exists(): print(f"stdout: {rr.out_file}")
        if rr.err_file and rr.err_file.exists(): print(f"stderr: {rr.err_file}")
        return

    if args.write_only:
        script = write_script(scheduler, spec)
        print(f"Wrote submission script: {script}")
        print("Submit manually with:", f"bsub < {script}" if scheduler == ScheduleKind.LSF else f"sbatch {script}")
        return

    jobid = submit(spec, scheduler)
    if args.submit_only:
        return  # job continues on cluster

    rr = monitor(scheduler, spec, jobid, args.poll)
    print(f"Job {rr.jobid} finished: {rr.final_state}")
    if rr.out_file and rr.out_file.exists(): print(f"stdout: {rr.out_file}")
    if rr.err_file and rr.err_file.exists(): print(f"stderr: {rr.err_file}")

if __name__ == '__main__':
    main()