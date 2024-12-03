import os 
from constructs import Construct 
from aws_cdk import App, Stack,Duration,CfnOutput,Environment
from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    # EcrImageCode,
    Architecture,
    FunctionUrlAuthType
)
from aws_cdk.aws_ecr import Repository

class GradioLambdaFn(Stack):
    def __init__(self,scope:Construct,construct_id:str,**kwargs)->None:
        super().__init__(scope,construct_id,**kwargs)
        

        # repo = Repository.from_repository_name(self,"ECRRepo",repository_name='mhema/dogsbreedclassifier') 

        lambda_fn = DockerImageFunction(
                            self,
                            id="DogBreedsDockerClassifier",
                            code=DockerImageCode.from_image_asset( directory= os.path.dirname(__file__) , file="Dockerfile"),
                            # code = DockerImageCode.from_ecr(
                            #     repository= repo,
                            #     tag_or_digest="latest"    #tag is depreciated
                            # ),
                            # code= DockerImageCode.from_ecr(repository=repo,tag_or_digest="latest"),
                            architecture=Architecture.X86_64,
                            memory_size=8192, #8GB
                            timeout=Duration.minutes(10),
                            description="dockerfile_lambda_deploy_using_cdk",
        )

        # HTTP URL add
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        # print
        CfnOutput(self,id="functionURL",value=fn_url.url,description="cfn_output")



env = Environment(
    account= os.environ.get('CDK_DEFAULT_ACCOUNT'),
    region=os.environ.get('CDK_DEFAULT_REGION','ap-south-1')
)

app = App()
gradio_lambda = GradioLambdaFn(app,"GradioLambda",env=env)
app.synth()