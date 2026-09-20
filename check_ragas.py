import sys
sys.stdout.reconfigure(encoding='utf-8')
from app.evaluation.ragas_compat import install_legacy_vertexai_import
install_legacy_vertexai_import()
from ragas.llms.base import InstructorLLM, InstructorModelArgs
import inspect
print(inspect.getsource(InstructorLLM._map_openai_params))
print("---")
# Check what copy() returns on InstructorModelArgs
args = InstructorModelArgs(max_tokens=2048)
print(type(args), dir(args))
