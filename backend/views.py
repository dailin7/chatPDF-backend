from rest_framework.decorators import api_view
from rest_framework.response import Response
from .chain import Chain
from rest_framework import status

chain = Chain()
chain.load_chain()

@api_view(["POST"])
def hello(request):
    return Response("Hello World")

@api_view(["POST"])
def query(request):
    try:
        query = request.data["query"]
        answer = chain.qa({"question": query})
        print(answer["answer"])
        return Response(answer["answer"], status=status.HTTP_200_OK, headers={"Access-Control-Allow-Origin": "*"},)
    except:
        return Response(request.data, status=status.HTTP_400_BAD_REQUEST)
    
@api_view(["POST"])
def upload_pdfs(request):
    try:
        question = request.data["query"]
        answer = chain.qa({"question": question})
        return Response(answer, status=status.HTTP_200_OK)
    except:
        return Response(request.data, status=status.HTTP_400_BAD_REQUEST)



# TEMPLATE
# @api_view(["GET", "POST"])
# def drink_list(request):
#     if request.method == "GET":
#         drinks = Drink.objects.all()
#         serializer = DrinksSerializer(drinks, many=True)
#         return Response(serializer.data, status=status.HTTP_200_OK)
    
#     if request.method == "POST":
#         serializer = DrinksSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
        
