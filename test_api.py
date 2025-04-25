import requests
import argparse
import os

# Base URL of the local API
default_url = os.getenv('API_URL', 'http://127.0.0.1:5000')


def login(username, password):
    url = f"{default_url}/login"
    resp = requests.post(url, auth=(username, password))
    print(resp.status_code, resp.text)
    resp.raise_for_status()
    data = resp.json()
    print("Login successful. Token:", data['token'])
    return data['token']


def register(token, image_path):
    url = f"{default_url}/attendance/register"
    headers = {'Authorization': f'Bearer {token}'}
    files = {'image': open(image_path, 'rb')}
    resp = requests.post(url, headers=headers, files=files)
    print('Register Response:', resp.status_code, resp.text)


def verify(token, image_path, latitude, longitude, location_id=None, pin_code=None, device_id=None):
    url = f"{default_url}/attendance/verify"
    headers = {'Authorization': f'Bearer {token}'}
    files = {'image': open(image_path, 'rb')}
    data = {'latitude': latitude, 'longitude': longitude}
    if location_id:
        data['location_id'] = location_id
    if pin_code:
        data['pin_code'] = pin_code
    if device_id:
        data['device_id'] = device_id
    resp = requests.post(url, headers=headers, files=files, data=data)
    print('Verify Response:', resp.status_code, resp.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Attendance Verification API')
    parser.add_argument('action', choices=['login', 'register', 'verify'], help='Action to perform')
    parser.add_argument('--username', help='Username for login')
    parser.add_argument('--password', help='Password for login')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--latitude', type=float, help='Latitude for verification')
    parser.add_argument('--longitude', type=float, help='Longitude for verification')
    parser.add_argument('--location_id', help='Location ID for verification')
    parser.add_argument('--pin_code', help='PIN code for verification')
    parser.add_argument('--device_id', help='Device ID for verification')
    args = parser.parse_args()

    if args.action == 'login':
        if not args.username or not args.password:
            print('Username and password are required for login')
        else:
            login(args.username, args.password)
    else:
        if not args.username or not args.password:
            print('Username and password are required')
            exit(1)
        token = login(args.username, args.password)
        if args.action == 'register':
            if not args.image:
                print('Image path is required for register')
            else:
                register(token, args.image)
        elif args.action == 'verify':
            if not args.image or args.latitude is None or args.longitude is None:
                print('Image path, latitude, and longitude are required for verify')
            else:
                verify(token, args.image, args.latitude, args.longitude, args.location_id, args.pin_code, args.device_id)
