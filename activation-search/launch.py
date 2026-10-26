"""Launch approved isolated activation-search workers under existing fleet leases."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,shlex,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'persona'))
from persona_launch import command

def runtime_paths(config):
    """Derive shared project assets from the isolated runtime's parent directory."""
    def absolute(value):
        if not isinstance(value,str) or not value.startswith('/') or any(ord(c)<32 for c in value) or '..' in PurePosixPath(value).parts:
            raise ValueError('Runtime asset paths must be absolute POSIX paths without traversal')
        return str(PurePosixPath(value))
    remote=absolute(config['remote_root'])
    project=absolute(config.get('remote_project_root',str(PurePosixPath(remote).parent)))
    return {'heliopause':absolute(config.get('heliopause_path',project+'/flare_heliopause.so')),
            'model':absolute(config.get('model_path',project+'/data/persona_model.json')),
            'calibration_dir':absolute(config.get('calibration_dir',project+'/results'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=int,default=90);parser.add_argument('--lane',choices=('interactive','batch'),default='interactive');parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'persona_fleet.local.json');parser.add_argument('--jobs-config',type=Path,default=Path(__file__).resolve().parents[1]/'persona_jobs.local.json');args=parser.parse_args()
    if not 30<=args.minutes<=600:raise ValueError('Session duration must be 30–600 minutes')
    root=Path(__file__).resolve().parent
    config=json.loads(args.config.read_text())
    paths=runtime_paths(config)
    settings=json.loads(args.jobs_config.read_text())
    ssh=['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',config['ssh_host']]
    def remote(argv):return subprocess.check_output([*ssh,shlex.join(argv)],text=True)
    assert remote(['hostname']).strip()==config['hostname']
    for p in root.glob('*.py'):
     if p.name=='launch.py':continue
     path=config['remote_root']+'/'+p.name
     assert remote(['sha256sum',path]).split()[0]==hashlib.sha256(p.read_bytes()).hexdigest(),p.name
    tunnel='character-activation-search-tunnel.service'
    if subprocess.run(['systemctl','--user','is-active','--quiet',tunnel]).returncode:
     tunnel_command=['systemd-run','--user','--unit='+tunnel,'--collect',f'--property=RuntimeMaxSec={args.minutes*60}','/usr/bin/ssh','-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']
     for port in range(8940,8946):tunnel_command += ['-L',f'127.0.0.1:{port}:127.0.0.1:{port}']
     subprocess.run(tunnel_command+[config['ssh_host']],check=True)
    for shard in range(6):
     unit=f'character-activation-search-probe-{shard}.service'
     check=subprocess.run([*ssh,shlex.join(['systemctl','--user','is-active','--quiet',unit])])
     if check.returncode==0:
      print('Existing isolated worker',shard,'left running',flush=True);continue
     invocation=command(config,shard,False,mode='probe',minutes=args.minutes,base_port=8940)
     unit=f'character-activation-search-probe-{shard}.service'
     invocation=[arg.replace(f'--unit=character-persona-probe-{shard}',f'--unit={unit}') for arg in invocation]
     invocation[invocation.index(config['lease_wrapper'])]=config['remote_root']+'/persona_lease.py'
     invocation[invocation.index('--output')+1]=config['remote_root']
     invocation[invocation.index('--heliopause')+1]=paths['heliopause']
     invocation[invocation.index('persona-probe-'+str(shard))]='activation-search-probe-'+str(shard)
     invocation += ['--model',paths['model'],
                    '--calibration',paths['calibration_dir']+f'/capture-shard{shard}.npz',
                    '--capture-module',config['remote_root']+'/activation_capture.py']
     at=invocation.index('/usr/bin/python3')
     invocation[at:at]=['--setenv=PERSONA_WORKER_UNIT='+unit,'--setenv=PERSONA_WORKBOARD_LANE='+args.lane,'--setenv=PERSONA_LEASE_WRAPPER='+config['lease_wrapper'],
                        '--setenv=PERSONA_LEASE_WRAPPER_SHA256='+settings['lease_wrapper_sha256']]
     print(remote(invocation),flush=True)
     print('Started isolated shard',shard,flush=True)

if __name__=='__main__':
    main()
