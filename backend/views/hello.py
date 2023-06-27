from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .. import sources
from qdrant_client.models import Distance, VectorParams
from ..utils import load_files


# chain = Chain()
# chain.load_chain()


@api_view(["POST"])
def hello(request):
    return Response("Hello World")


@api_view(["POST"])
def query(request):
    try:
        query = request.data["query"]
        answer = ""  # chain.qa({"question": query})
        print(answer["answer"])
        return Response(
            answer["answer"],
            status=status.HTTP_200_OK,
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except:
        return Response(request.data, status=status.HTTP_400_BAD_REQUEST)
