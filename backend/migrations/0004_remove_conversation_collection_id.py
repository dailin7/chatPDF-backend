# Generated by Django 4.2.1 on 2023-06-27 09:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0003_alter_collection_id_alter_conversation_id_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='conversation',
            name='collection_id',
        ),
    ]
