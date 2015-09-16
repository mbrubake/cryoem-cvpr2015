import subprocess
import shlex

def git_info_dump(outname):
	with open(outname, 'w') as f:
		cmd = 'git rev-parse HEAD'
		args = shlex.split(cmd)
		subprocess.call(args, stdout=f)	
		cmd = 'git status'
		args = shlex.split(cmd)
		subprocess.call(args, stdout=f)	
		cmd = 'git diff'
		args = shlex.split(cmd)
		subprocess.call(args, stdout=f)	

def git_get_SHA1():
	args = shlex.split('git rev-parse HEAD')
	return subprocess.check_output(args).strip()