"""Windows resource sampling and first-cause stop requests; standard library only."""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent

def settings():
    return json.loads((ROOT/'settings.json').read_text(encoding='utf-8-sig'))

def first_stop(path, reason):
    try:
        with Path(path).open('x',encoding='utf-8') as stream:stream.write(reason)
    except FileExistsError:
        pass

def sample():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
            ('total_ram','free_ram','total_commit','free_commit','total_virtual','free_virtual','extended_virtual')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW)
    lines=query.stdout.strip().splitlines()
    if len(lines)!=1:raise RuntimeError('Expected exactly one reviewed GPU')
    used,free,utilization,power,temperature=map(float,lines[0].split(','))
    return {'at':datetime.now(timezone.utc).isoformat(),'free_commit_gib':memory.free_commit/1024**3,
        'free_ram_gib':memory.free_ram/1024**3,'gpu_used_mib':used,'gpu_free_mib':free,
        'gpu_utilization_percent':utilization,'gpu_power_watts':power,'gpu_temperature_c':temperature}

def resource_reason(row, limits):
    if row['free_commit_gib']<limits['stop_free_commit_gib']:return 'host_memory_guard'
    if row['gpu_free_mib']<limits['stop_free_gpu_mib']:return 'gpu_memory_guard'
    if row['gpu_temperature_c']>=limits['stop_gpu_temperature_c']:return 'gpu_temperature_guard'
    if row['free_ram_gib']<limits['critical_free_ram_gib']:return 'critical_physical_memory_guard'
    if row['free_ram_gib']<limits['stop_free_ram_gib']:return 'physical_memory_guard'
    return None

def owned_collector_alive(pid, run_id):
    # Require both the exact source path and unique run id, not PID existence.
    command=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; if ($p) {{ $p.CommandLine }}"
    result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,
        timeout=15,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return str(ROOT/'collector.py').casefold() in result.stdout.casefold() and run_id in result.stdout


class ResourceGuard:
    """Keep hard limits immediate; allow a bounded recovery window for soft RAM pressure."""
    def __init__(self, limits):
        self.limits=limits
        self.physical_low_since=None

    def check(self, row, now):
        reason=resource_reason(row,self.limits)
        if reason!='physical_memory_guard':
            self.physical_low_since=None
            return reason
        if self.physical_low_since is None:
            self.physical_low_since=now
        if now-self.physical_low_since>=self.limits['physical_memory_grace_seconds']:
            return reason
        return None
