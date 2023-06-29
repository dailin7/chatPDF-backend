"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from backend.views import hello, user, conversation, collection as db_collection
from backend.vectorDB import collection

urlpatterns = [
    path("admin/", admin.site.urls),
    path("query/", hello.query),
    path("hello/", hello.hello),
    path("collection/create", collection.create_collection), #for creating vector db collection
    path("collection/names", collection.get_names), #for retrieveing vector db collections' names
    path("collection/<str:collection_name>", collection.get_collection),
    path("collection/<str:collection_name>/delete", collection.delete_collection),
    path("collection/<str:collection_name>/upload", collection.upload_files), #upload files to existing db collection
    path("user/create", user.create_user),
    path("conversation/create", conversation.create_conversation), #for creating conversation
    path("conversation/<str:user_id>", conversation.get_conversations_by_user_id),
    path("db_collection/create", db_collection.create_collection),
]
