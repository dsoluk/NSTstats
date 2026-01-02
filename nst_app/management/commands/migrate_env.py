import os
from django.core.management.base import BaseCommand
from nst_app.models import Parameter
from dotenv import load_dotenv

class Command(BaseCommand):
    help = 'Migrates parameters from .env to the database'

    def handle(self, *args, **options):
        # We don't want to use load_dotenv() directly because we want to parse the file
        # to distinguish between secrets and normal parameters if possible, 
        # but the user said everything except YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET are parameters.
        
        secrets = ['YAHOO_CLIENT_ID', 'YAHOO_CLIENT_SECRET']
        
        if not os.path.exists('.env'):
            self.stdout.write(self.style.ERROR('.env file not found'))
            return

        with open('.env', 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                if key in secrets:
                    self.stdout.write(self.style.SUCCESS(f'Skipping secret: {key}'))
                    continue
                
                param, created = Parameter.objects.update_or_create(
                    key=key,
                    defaults={'value': value}
                )
                if created:
                    self.stdout.write(self.style.SUCCESS(f'Created parameter: {key}'))
                else:
                    self.stdout.write(self.style.SUCCESS(f'Updated parameter: {key}'))
