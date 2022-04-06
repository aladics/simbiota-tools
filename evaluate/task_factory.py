from tests.res.runner import LearnTask
from shutil import rmtree
from pathlib import Path

from common import util
util.set_dbh_path()

from tests.res.runner import LearnTask
from tests.res import params

class TaskFactory:
    class DNNCLearnTask(LearnTask):
        def __init__(self, shared_params, sandbox : str):
            super().__init__(shared_params)
            self.sandbox = Path(sandbox)

        def pre_run(self):
            self.sandbox.mkdir(parents=True, exist_ok=True)

        def post_run(self):
            if Path(self.sandbox).exists():
                rmtree(str(self.sandbox.resolve()))
               
    def convert_param(self, param):
        try:
            param = int(param)
        except ValueError:
            try:
                param = float(param)
            except ValueError:
                pass
    
        return param
    
    def parse_model(self, model_spec : str):
        model_spec_split = model_spec.split(" ")
        model_name = model_spec_split[0]
        
        args = {}
        
        for i in range(1, len(model_spec_split) -1, 2):
            param_name = model_spec_split[i].replace("--", "").replace("-", "_")
            args[param_name] = self.convert_param(model_spec_split[i+1])
        
        return model_name, args
                
    def __init__(self, shared_params):
        self.shared_params = shared_params

    def get(self, model_spec, task_id):
        model_name, args = self.parse_model(model_spec)
        
        if model_name != "cdnnc" and model_name != "sdnnc":
            learn_task = LearnTask(self.shared_params)
        else:
            #TODO: !!
            config = util.get_config()
            
            sandbox_path = Path(config['sandbox_root']) / task_id
            sandbox_path_str = str(sandbox_path.resolve())
            args['sandbox'] = sandbox_path_str

            learn_task = self.DNNCLearnTask(self.shared_params, sandbox_path_str)
       
        model_params = None
        if model_name == "svm":
            if not args:
                model_params =  params.SVMParams()                 
            else:
                model_params =  params.SVMParams(**args)            
        elif model_name == "forest":
            if not args:
                model_params =  params.ForestParams()
            else:
                model_params =  params.ForestParams(**args)
        elif model_name == "sdnnc":
            if not args:
                model_params =  params.SDNNCParams()
            else:
                model_params = params.SDNNCParams(**args)
        elif model_name == "cdnnc":
            if not args:
                model_params =  params.CDNNCParams()
            else:
                model_params = params.CDNNCParams(**args)
        elif model_name == "adaboost":
            if not args:
                model_params =  params.AdaboostParams()
            else:
                model_params = params.AdaboostParams(**args)
        elif model_name == "knn":
            if not args:
                model_params =  params.KNNParams()
            else:
                model_params = params.KNNParams(**args)
        elif model_name == "logistic":
            if not args:
                model_params =  params.LogisticParams()
            else:
                model_params = params.LogisticParams(**args)
        elif model_name == "linear":
            if not args:
                model_params =  params.LinearParams()
            else:
                model_params = params.LinearParams(**args)
        elif model_name == "tree":
            if not args:
                model_params =  params.TreeParams()
            else:
                model_params = params.TreeParams(**args)
                    
            
        learn_task.params = model_params.get()
        learn_task.set_save_model_path("")

        return learn_task