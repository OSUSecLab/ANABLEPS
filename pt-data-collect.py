import os, commands
import sys, time
import argparse

num_testing = 10

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

	# Get inputs
	# python pt-data-collect.py -b ls -a "-l %s" -o [path to store pt data] -i [path of generated inputs]
	parser = argparse.ArgumentParser(prog="pt-data-collect.py", description='Retrieve branch running time, PT packet list for a binary with specific input\nThis file need ROOT priviledge')
	parser.add_argument('-b', dest='bin_path', action='store', required=True, help='The binary to be run')
	parser.add_argument('-a', dest='bin_args', action='store', help='The arguments for the binary')
	parser.add_argument('-o', dest='output_path', action='store', help='The path of output files')
	parser.add_argument('-p', dest='perf_path', action='store', help='The path of perf program')
	parser.add_argument('-i', dest='input_file_path', action='store', help='The input file path')
	parser.add_argument('-f', dest='is_file', default=True, type=str2bool, action='store', help='The input type is the file, or the content in the file')
	args = parser.parse_args()
	file_name = os.path.basename(args.bin_path)
	if args.output_path is None:
		if not os.path.exists('./pt_data'):
			os.makedirs('./pt_data')
		args.output_path = os.path.join('./pt_data', file_name + '_perfdata')
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)

	# Disable ASLR
	s, echo = commands.getstatusoutput('echo 0 | sudo tee /proc/sys/kernel/randomize_va_space')
	if echo != '0':
		print 'Disable Randomize Failed %s' %echo
	s, echo = commands.getstatusoutput('echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope')
	if echo != '0':
		print 'Disable Ptrace Failed %s' %echo

	if args.input_file_path is not None:
		files = os.listdir(args.input_file_path)
		for i in range(num_testing):
			for file in files:
				if not os.path.exists(os.path.join(args.output_path,file)):
					os.makedirs(os.path.join(args.output_path,file))
				if not args.is_file:
					with open(os.path.join(args.input_file_path, file)) as f:
						l = f.readline().strip()
					cmd = 'sudo %s record -e intel_pt/tsc=1,mtc=1,mtc_period=0,cyc=1,cyc_thresh=0/u %s %s ' %(args.perf_path, args.bin_path, args.bin_args %l)
				else:
					cmd = 'sudo %s record -e intel_pt/tsc=1,mtc=1,mtc_period=0,cyc=1,cyc_thresh=0/u %s %s ' %(args.perf_path, args.bin_path, args.bin_args %os.path.join(args.input_file_path,file))
				print '\nRunning: %s' %cmd
				os.system(cmd) 
				cmd = 'sudo cp ./perf.data %s' %os.path.join(args.output_path,file, str(i+1)+'th_perf.data')
				os.system(cmd)
				time.sleep(0.1)
	else:
		if not os.path.exists(os.path.join(args.output_path,'no-input-file')):
			os.makedirs(os.path.join(args.output_path,'no-input-file'))
		for i in range(num_testing):
				cmd = 'sudo %s record -o %s -e intel_pt/tsc=1,mtc=1,mtc_period=0,cyc=1,cyc_thresh=0/u %s %s ' %(args.perf_path, os.path.join(args.output_path,'no-input-file', str(i+1)+'th_perf.data'), args.bin_path, args.bin_args)
				print '\nRunning: %s' %cmd
				os.system(cmd) 

