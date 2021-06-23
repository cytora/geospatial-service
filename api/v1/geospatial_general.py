import psycopg2
from psycopg2 import Error

from fastapi import APIRouter
from typing import Optional

import logging
import os
import time
from dotenv import load_dotenv

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
env = load_dotenv()

router = APIRouter()


def select_query_dict(connection, query, data=[]):
    """
    Run generic select query on db, returns a list of dictionaries
    """
    logger.debug('Running query: {}'.format(query))

    # Open a cursor to perform database operations
    cursor = connection.cursor()
    logging.debug('Db connection succesful')

    # execute the query
    try:
        logger.info('Running query.')
        if len(data):
            cursor.execute(query, data)
        else:
            cursor.execute(query)
        columns = list(cursor.description)
        result = cursor.fetchall()
        logging.debug('Query executed succesfully')
    except (Exception, psycopg2.DatabaseError) as e:
        logging.error(e)
        cursor.close()
        exit(1)

    cursor.close()

    # make dict
    results = []
    for row in result:
        row_dict = {}
        for i, col in enumerate(columns):
            row_dict[col.name] = row[i]
        results.append(row_dict)

    return results


class PostgresConfiguration():
    POSTGRESQL_DB_PORT = os.getenv('POSTGRES_PORT', 5432)
    POSTGRESQL_DB_NAME = os.getenv('POSTGRES_DB', 'cytora_data_rds')
    POSTGRESQL_DB_USER = os.getenv('POSTGRES_USER', 'geo')
    POSTGRESQL_DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'TGL2022!!')
    POSTGRESQL_DB_HOST = os.getenv('POSTGRES_ADDRESS', 'dev-postgres-11.c5xohzyav5el.eu-west-1.rds.amazonaws.com')

    @property
    def postgres_db_path(self):
        return f'postgresql://{self.POSTGRESQL_DB_USER}:{self.POSTGRESQL_DB_PASSWORD}@' \
               f'{self.POSTGRESQL_DB_HOST}:' \
               f'{self.POSTGRESQL_DB_PORT}/{self.POSTGRESQL_DB_NAME}'

    @property
    def pg2(self):
        return psycopg2.connect(
            user=self.POSTGRESQL_DB_USER,
            password=self.POSTGRESQL_DB_PASSWORD,
            host=self.POSTGRESQL_DB_HOST,
            port=self.POSTGRESQL_DB_PORT,
            database=self.POSTGRESQL_DB_NAME
        )


@router.get('/discovery/data')
async def get_discovery():
    '''
    purpose: function discover available GeoSpatial Layers/Tables for query/search.
    get all tables with GEOM column in public Schema and respond back with object as follows:

    {
        "layers": [
            {
                "gis_layer": "geo_uk_haz_t100_03",
                "srid": 4326,
                "count": 1355
            },
            {
                "gis_layer": "geo_uk_haz_t10_03",
                "srid": 4326,
                "count": 12586
            },
            {
                "gis_layer": "geo_uk_haz_t5_03",
                "srid": 4326,
                "count": 24585
            }
        ],
        "exec_time_seconds": "2.403485804"
    }
    SRID stands for Spatial Reference ID. 4326 => WGS84, 27770 => UK GRID, ...
    COUNT presents number of objects/rows in given table/layer
    '''

    sql = '''
        select
            table_name as gis_layer
        from information_schema.columns
        where table_schema not in ('information_schema', 'pg_catalog') and column_name = 'geom' and table_schema = 'public'
        order by 
            table_schema, 
            table_name,
            ordinal_position;
    '''
    start = time.perf_counter()
    with PostgresConfiguration().pg2 as con:
        res = select_query_dict(con, sql)

    for el in res:
        print(el)
        with PostgresConfiguration().pg2 as con:
            cur = con.cursor()

            srid = f'''SELECT Find_SRID('public', '{el['gis_layer']}', 'geom');'''
            cur.execute(srid)
            srd = cur.fetchall()

            count = f'''SELECT count(1) from {el['gis_layer']};'''
            cur.execute(count)
            cnt = cur.fetchall()

            el['srid'] = srd[0][0]
            el['count'] = cnt[0][0]

    obj = {}
    obj['layers'] = res
    obj['exec_time_seconds'] = f'{time.perf_counter() - start}'
    return obj


@router.get('/intersect/{lat}/{lon}/{lyr}')
async def get_intersection(lat: float, lon: float, lyr: str):
    '''
    purpose: Find Intersection/drill down between caller provided lat, lon and feature layer name
    example

    request URI => http://.../v1/intersect/52.71/-1.82/geo_uk_haz_t10_03

    response =>
    {
        "request": {
            "lat": 52.71,
            "lon": -1.82,
            "layer": "geo_uk_haz_t10_03"
        },
        "response": [
            {
                "id": 2558,
                "t10_id": "10_1_2558",
                "country": "Great Britain",
                "area_km2": 19
            }
        ],
        "exec_time_seconds": "0.6775946429999635"
    }
    '''
    sql = f'''
    SELECT
    *
    FROM {lyr}
    WHERE ST_Intersects(geom, 'SRID=4326;POINT({lon} {lat})');
    '''
    start = time.perf_counter()
    with PostgresConfiguration().pg2 as con:
        res = select_query_dict(con, sql)

    for el in res:
        del el['geom']

    obj = {}
    obj['request'] = {'lat': lat, 'lon': lon, 'layer': lyr}
    obj['response'] = res
    obj['exec_time_seconds'] = f'{time.perf_counter() - start}'
    return obj
