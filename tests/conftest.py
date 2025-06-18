import sys
from pathlib import Path
import pytest
import uuid
from google.cloud import storage

# Ensure the src directory is on sys.path for tests
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

@pytest.fixture(scope="session")
def gcs_test_bucket():
    """
    Pytest fixture to create and manage a temporary GCP bucket for testing.
    Uploads dummy data and cleans up after tests.
    """
    bucket_name = f"f1-data-for-chatbot-test-{uuid.uuid4()}"
    client = storage.Client()
    bucket = client.create_bucket(bucket_name)

    # Dummy data for testing
    drivers_csv_content = """driverId,driverRef,number,code,forename,surname,dob,nationality,url
1,hamilton,44,HAM,Lewis,Hamilton,1985-01-07,British,http://en.wikipedia.org/wiki/Lewis_Hamilton
2,alonso,14,ALO,Fernando,Alonso,1981-07-29,Spanish,http://en.wikipedia.org/wiki/Fernando_Alonso
12,Kimi ANTONELLI,Kimi,Antonelli,ANT,Mercedes,00D7B6,,K ANTONELLI,9971,1262
63,George RUSSELL,George,Russell,RUS,Mercedes,00D7B6,,G RUSSELL,9971,1262
5,Gabriel BORTOLETO,Gabriel,Bortoleto,BOR,Kick Sauber,01C00E,,G BORTOLETO,9971,1262
27,Nico HULKENBERG,Nico,Hulkenberg,HUL,Kick Sauber,01C00E,,N HULKENBERG,9971,1262
23,Alexander ALBON,Alexander,Albon,ALB,Williams,1868DB,,A ALBON,9971,1262
55,Carlos SAINZ,Carlos,Sainz,SAI,Williams,1868DB,,C SAINZ,9971,1262
14,Fernando ALONSO,Fernando,Alonso,ALO,Aston Martin,229971,,F ALONSO,9971,1262
1,Max VERSTAPPEN,Max,Verstappen,VER,Red Bull Racing,4781D7,,M VERSTAPPEN,9971,1262
22,Yuki TSUNODA,Yuki,Tsunoda,TSU,Red Bull Racing,4781D7,,Y TSUNODA,9971,1262
6,Isack HADJAR,Isack,Hadjar,HAD,Racing Bulls,6C98FF,,I HADJAR,9971,1262
30,Liam LAWSON,Liam,Lawson,LAW,Racing Bulls,6C98FF,,L LAWSON,9971,1262
31,Esteban OCON,Esteban,Ocon,OCO,Haas F1 Team,9C9FA2,,E OCON,9971,1262
87,Oliver BEARMAN,Oliver,Bearman,BEA,Haas F1 Team,9C9FA2,,O BEARMAN,9971,1262
16,Charles LECLERC,Charles,Leclerc,LEC,Ferrari,ED1131,,C LECLERC,9971,1262
44,Lewis HAMILTON,Lewis,Hamilton,HAM,Ferrari,ED1131,,L HAMILTON,9971,1262
4,Lando NORRIS,Lando,Norris,NOR,McLaren,F47600,,L NORRIS,9971,1262
81,Oscar PIASTRI,Oscar,Piastri,PIA,McLaren,F47600,,O PIASTRI,9971,1262"""

    lap_times_csv_content = """raceId,driverId,lap,position,time,milliseconds
1,1,1,1,1:30.000,90000"""

    bucket.blob("drivers_F1.csv").upload_from_string(drivers_csv_content)
    bucket.blob("lap_times_F1.csv").upload_from_string(lap_times_csv_content)

    yield bucket_name

    # Teardown: delete all blobs and then the bucket
    bucket.delete(force=True)
