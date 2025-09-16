import os
import hmac
import hashlib
import base64
import json
import time
from dotenv import load_dotenv

# Conditionally import boto3
try:
    import boto3
    from botocore.exceptions import ClientError
    boto3_available = True
except ImportError:
    boto3_available = False

# Load environment variables
load_dotenv()

# These would be set in environment variables in production
# For local development, they're placeholders
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
COGNITO_USER_POOL_ID = os.environ.get('COGNITO_USER_POOL_ID', '')
COGNITO_APP_CLIENT_ID = os.environ.get('COGNITO_APP_CLIENT_ID', '')
COGNITO_APP_CLIENT_SECRET = os.environ.get('COGNITO_APP_CLIENT_SECRET', '')

# Initialize the Cognito Identity Provider client
cognito_client = None
if boto3_available:
    try:
        cognito_client = boto3.client('cognito-idp', region_name=AWS_REGION)
    except Exception as e:
        print(f"Warning: Could not initialize Cognito client: {str(e)}")
        cognito_client = None

def local_auth_enabled():
    """Check if local authentication should be used"""
    return not cognito_client or not COGNITO_USER_POOL_ID or not COGNITO_APP_CLIENT_ID

def get_secret_hash(username):
    """Generate a secret hash for the Cognito API"""
    if not COGNITO_APP_CLIENT_SECRET:
        return None
        
    message = username + COGNITO_APP_CLIENT_ID
    dig = hmac.new(
        COGNITO_APP_CLIENT_SECRET.encode('utf-8'),
        msg=message.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(dig).decode()

def sign_up(username, password, email):
    """Register a new user with Cognito"""
    if local_auth_enabled():
        print("Using local authentication.")
        # For local development, just return success
        return {
            "success": True,
            "message": "Local user registration successful",
            "user_id": username
        }
    
    try:
        secret_hash = get_secret_hash(username)
        params = {
            'ClientId': COGNITO_APP_CLIENT_ID,
            'Username': username,
            'Password': password,
            'UserAttributes': [
                {'Name': 'email', 'Value': email}
            ]
        }
        
        if secret_hash:
            params['SecretHash'] = secret_hash
            
        response = cognito_client.sign_up(**params)
        
        return {
            "success": True,
            "message": "User registration successful",
            "user_id": response['UserSub']
        }
    except ClientError as e:
        return {
            "success": False,
            "message": str(e)
        }

def confirm_sign_up(username, confirmation_code):
    """Confirm a user registration with the verification code"""
    if local_auth_enabled():
        print("Using local authentication.")
        # For local development, just return success
        return {
            "success": True,
            "message": "Local user confirmation successful"
        }
    
    try:
        secret_hash = get_secret_hash(username)
        params = {
            'ClientId': COGNITO_APP_CLIENT_ID,
            'Username': username,
            'ConfirmationCode': confirmation_code
        }
        
        if secret_hash:
            params['SecretHash'] = secret_hash
            
        cognito_client.confirm_sign_up(**params)
        
        return {
            "success": True,
            "message": "User confirmation successful"
        }
    except ClientError as e:
        return {
            "success": False,
            "message": str(e)
        }

def sign_in(username, password):
    """Authenticate a user and get tokens"""
    if local_auth_enabled():
        print("Using local authentication.")
        # For development, just return the username as the user_id
        if username:
            return {
                "success": True,
                "message": "Local authentication successful",
                "user_id": username,
                "tokens": {"id_token": "local_id_token", "access_token": "local_access_token"}
            }
        else:
            return {"success": False, "message": "Username required"}
    
    try:
        secret_hash = get_secret_hash(username)
        params = {
            'ClientId': COGNITO_APP_CLIENT_ID,
            'AuthFlow': 'USER_PASSWORD_AUTH',
            'AuthParameters': {
                'USERNAME': username,
                'PASSWORD': password
            }
        }
        
        if secret_hash:
            params['AuthParameters']['SECRET_HASH'] = secret_hash
            
        response = cognito_client.initiate_auth(**params)
        
        return {
            "success": True,
            "message": "Authentication successful",
            "user_id": username,
            "tokens": {
                "id_token": response['AuthenticationResult']['IdToken'],
                "access_token": response['AuthenticationResult']['AccessToken'],
                "refresh_token": response['AuthenticationResult']['RefreshToken']
            }
        }
    except ClientError as e:
        return {
            "success": False,
            "message": str(e)
        }

def verify_token(token):
    """Verify a JWT token from Cognito"""
    if local_auth_enabled():
        print("Using local authentication.")
        # For development, just return 
        
        # In a real implementation, we would extract the username from the token
        return {"success": True, "username": "local_user"}
    
    try:
        response = cognito_client.get_user(
            AccessToken=token
        )
        
        # Extract username from response
        username = response['Username']
        
        return {
            "success": True,
            "username": username
        }
    except ClientError as e:
        return {
            "success": False,
            "message": str(e)
        }

# For local development without AWS
def local_auth_enabled():
    """Check if we're using local authentication"""
    return cognito_client is None
