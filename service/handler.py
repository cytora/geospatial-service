# pylint: skip-file
"""REST wrapper for resolve service"""
import asyncio
from typing import Any, Dict, Optional

# import py_platform_utils.bi_publisher as bi
from py_platform_utils.config import Config
from py_platform_utils.logging import Log
from py_platform_utils.server.sanic_app import create_sanic_app
from sanic.request import Request
from sanic.response import json

import psycopg2
from psycopg2 import Error

import logging
import os
import time


class PostgresConfiguration():
    POSTGRESQL_DB_PORT = os.getenv('POSTGRESQL_DB_PORT')
    POSTGRESQL_DB_NAME = os.getenv('POSTGRESQL_DB_NAME')
    POSTGRESQL_DB_USER = os.getenv('POSTGRES_DB_USER')
    POSTGRESQL_DB_PASSWORD = os.getenv('POSTGRES_DB_PASSWORD')
    POSTGRESQL_DB_HOST = os.getenv('POSTGRES_DB_HOST')

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


def select_query_dict(connection, query, data=[]):
    """
    Run generic select query on db, returns a list of dictionaries
    """
    #logger.debug('Running query: {}'.format(query))

    # Open a cursor to perform database operations
    cursor = connection.cursor()
    logging.debug('Db connection succesful')

    # execute the query
    try:
        #logger.info('Running query.')
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


# new sanic framework requirements -> provide name as prop to the create_sanic_app class/function
#app = create_sanic_app(name='universal_resolver')
app = create_sanic_app()
version = 'v1'
conf = Config()

# settings
threshold_default = conf.get_value('threshold', None)


@app.route(f'/{version}/discovery/layers', methods=['GET'])
async def get_discovery(request: Request) -> Dict[str, Any]:
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

    log: Log = request.get('log')

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
    try:
        with PostgresConfiguration().pg2 as con:
            res = select_query_dict(con, sql)

        for el in res:
            print(el)
            with PostgresConfiguration().pg2 as con:
                cur = con.cursor()

                srid = f'''SELECT Find_SRID('public', '{el['gis_layer']}', 'geom');'''
                cur.execute(srid)
                srd = cur.fetchall()

                #count = f'''SELECT count(1) from {el['gis_layer']};'''
                #cur.execute(count)
                #cnt = cur.fetchall()

                #geom_type = f'''SELECT count(1), ST_GeometryType(geom) as geom_type
                #    from {el['gis_layer']}
                #    group by ST_GeometryType(geom);'''
                #geom_typ = select_query_dict(con, geom_type)

                #ext_sql = f'''SELECT ST_AsGeoJSON(ST_Extent(geom)) as extent FROM {el['gis_layer']};'''
                #ext = select_query_dict(con, ext_sql)

                el['srid'] = srd[0][0]
                # el['count'] = cnt[0][0]
                #el['geometry'] = geom_typ[0]
                #d = ast.literal_eval(ext[0]['extent'])
                #el['extent'] = d
        print(type(res), res)
        return_payload = {
            'layers': res,
            'exec_time_seconds': f'{time.perf_counter() - start}'
        }

    except Exception as e:
        log.error("error in geospatial discovery", error=e)
        print(e)
        raise e
    finally:
        log.info(f'final messages here')
        #bi_message = generate_bi_message(body, response, log=log)
        #bi.publish(bi_message, trace_id=log.trace_id, is_test=headers_is_test(request.headers))

    return to_json(return_payload)


@app.route(f'/{version}/intersect', methods=['GET'])
async def get_intersect(request: Request) -> Dict[str, Any]:
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
    log: Log = request.get('log')
    layer = request.args.get('layer')
    latitude = request.args.get('latitude', 52.20)
    longitude = request.args.get('longitude', 0.00)

    sql = f'''
    SELECT
    -- ST_AsGeoJSON(geom) as g,
    *
    FROM {layer}
    WHERE ST_Intersects(geom, 'SRID=4326;POINT({longitude} {latitude})');
    '''
    start = time.perf_counter()
    return_payload = {}
    try:
        with PostgresConfiguration().pg2 as con:
            res = select_query_dict(con, sql)

        #geometry = None
        for el in res:
            #geometry = el['g']
            del el['geom']
            #del el['g']

        return_payload = {
            'request': {
                'lat': latitude,
                'lon': longitude,
                'layer': layer},
            'response': res,
            'exec_time_seconds': f'{time.perf_counter() - start}'
        }

    except Exception as e:
        log.error("error matching address", error=e)
        print(e)
        raise e
    finally:
        log.info(f'final messages here')
        #bi_message = generate_bi_message(body, response, log=log)
        #bi.publish(bi_message, trace_id=log.trace_id, is_test=headers_is_test(request.headers))

    print(type(return_payload), return_payload)
    return to_json(return_payload)


def to_json(obj) -> Dict[str, Any]:
    return json(obj)


# main
def run():
    """Serve the REST service"""
    app.run(host='0.0.0.0', port=3000, debug=False, access_log=False)


if __name__ == '__main__':
    print("building ...")
    if os.environ.get('local'):
        run()
