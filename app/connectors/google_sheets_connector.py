from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from app.connectors.base import BaseConnector

class GoogleSheetsConnector(BaseConnector):
    def __init__(self, credentials_file, spreadsheet_id):
        self.credentials_file = credentials_file
        self.spreadsheet_id = spreadsheet_id
        self.service = None

    def connect(self):
        creds = Credentials.from_authorized_user_file(self.credentials_file, ['https://www.googleapis.com/auth/spreadsheets'])
        self.service = build('sheets', 'v4', credentials=creds)

    def disconnect(self):
        if self.service:
            self.service.close()

    def query(self, range_name):
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=self.spreadsheet_id, range=range_name).execute()
        values = result.get('values', [])
        if not values:
            return []
        headers = values[0]
        return [dict(zip(headers, row)) for row in values[1:]]

    def insert(self, range_name, data):
        sheet = self.service.spreadsheets()
        values = [[data[header] for header in data.keys()]]
        body = {'values': values}
        sheet.values().append(
            spreadsheetId=self.spreadsheet_id, range=range_name,
            valueInputOption='USER_ENTERED', body=body).execute()

    def update(self, range_name, data):
        sheet = self.service.spreadsheets()
        values = [[data[header] for header in data.keys()]]
        body = {'values': values}
        sheet.values().update(
            spreadsheetId=self.spreadsheet_id, range=range_name,
            valueInputOption='USER_ENTERED', body=body).execute()

    def delete(self, range_name):
        sheet = self.service.spreadsheets()
        sheet.values().clear(spreadsheetId=self.spreadsheet_id, range=range_name).execute()
