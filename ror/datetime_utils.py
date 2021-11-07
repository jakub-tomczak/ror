import datetime


def get_date_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H-%M-%S")