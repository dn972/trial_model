import os
import multiprocessing
import concurrent.futures
import subprocess

class Ultility(object):
    @staticmethod
    def addParams(command: str, params: str, paramsContent: str):
        return command + " " + params + " " + paramsContent

    @staticmethod
    def cleanUp(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    @staticmethod
    def makedirs_ifnotexist(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def runIntensiveTask(cmdList: list, numConcurrentWorks=None):
        if numConcurrentWorks is None:
            numConcurrentWorks = os.cpu_count()
        print("Number of concurrent threads: ", numConcurrentWorks)
        print("Total commands have to be done: ", len(cmdList))
        print("Working......Please wait!")

        with concurrent.futures.ThreadPoolExecutor(numConcurrentWorks) as executor:
            future_to_url = [executor.submit(Ultility.__spawmProcess, cmd, output) for cmd, output in cmdList]
            # for future in concurrent.futures.as_completed(future_to_url):

        print("All worker threads have been finished!")
    @staticmethod
    def __spawmProcess(cmd, output):
        if output is None:
            output = os.devnull

        with open(output, 'w') as f:
            # print("Working command: ", " >> ", cmd, "\n")
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, shell=True)
        # groups = []
        # if hasStdOut:
        #     for cmd, output in cmdList:
        #         with open(output,'wt') as f:
        #             groups.append(subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, shell=True)
                
        #     groups = groups*numConcurrentWorks
        #     # groups = [(subprocess.Popen(cmd, stdout= (with open(output, 'wt') as f), stderr=subprocess.PIPE, shell=True)
        #     #        for cmd, output in cmdList)] * numConcurrentWorks  # itertools' grouper recipe
        # else:
        #     groups = [(subprocess.Popen(cmd, stdout=open(os.devnull, 'wt'), stderr=subprocess.PIPE, shell=True)
        #            for cmd in cmdList)] * numConcurrentWorks  # itertools' grouper recipe  
        # print(groups)
        # # run len(processes) == limit at a time
        # for processes in zip_longest(*groups):
        #     # for pe in processes:
        #     #     print(pe.args)
        #     for p in filter(None, processes):
        #         p.wait()
        
