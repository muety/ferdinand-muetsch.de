---
title: Serving raster data from PostGIS as WMS using Python and FastAPI or Flask
date: 2021-10-21 17:13:53
tags: [gis]
---

![](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/raster_data_2.jpg)

# Raster Data
When dealing with geo data, there are essentially two major types of data: vector data and raster data. Vector data includes points, lines and polygons, while raster data is represented as a two- or three-dimensional pixel map.

If your aim is to map every fast-food restaurant in the US, you would probably represent each of them as a point, i.e. you will deal with vector data. 

Raster data, on the other hand, could, for example, be the geographical altitude of every single pixel within some area. Or, in the case of [Orbio](https://orbio.earth) – where we largely work with satellite imagery – a satellite observation of a certain geographical area is also represented as raster of pixels, each of which holds, for instance, a value for the observed light intensity / brightness. 

![](https://apps.muetsch.io/images/o:auto/rs,s:640?image=https://muetsch.io/images/raster_data_1.png) (https://desktop.arcgis.com/de/arcmap/10.3/manage-data/raster-and-images/what-is-raster-data.htm)

Every raster has a certain spatial resolution, that is, every pixel represents a certain area in m². Moreover, raster can – but do not have to – be geo-referenced, i.e. aligned to some geographical coordinate system (the most prominent of which is `EPSG:4326`).

# PostGIS, Geoserver and WMS
The by far most popular way to store and query any kind of geo data is using [PostGIS](https://postgis.net/), which is a brilliant, open-source extension to the PostgreSQL database management system. It supports vector data types (e.g. `POINT`, `POLYGON`, `LINESTRING`) as well as raster data (`RASTER` type).

Besides storing the data, a common use case is to visualize it on a map. For web applications, popular map libraries include [OpenLayers](https://openlayers.org/) and [Leaflet](https://leafletjs.com/). To visualize it, these libraries first need to pull the geo data from somewhere. However, a client-side JavaScript application can (and should) not get direct access to the database. Instead, usually, there will be a backend-side web application in between, acting as an intermediary.

This is where [WMS](https://en.wikipedia.org/wiki/Web_Map_Service) (aka. _web map service_) comes in. WMS is a protocol, that specifies how to request geo data on the web. Input parameters include a bounding box of the requested area, an SRID code and others, while the output of a WMS service is rendered images (usually PNGs, JPEGs or TIFFs).

Consequently, as the developer of a [GIS](https://en.wikipedia.org/wiki/Geographic_information_system) application, you somehow need to provide a WMS endpoint, that, on one end, fetches data from PostGIS, and, on the other end, delivers it to the web frontend. 

One approach to do so is to use a map server, like [Geoserver](http://geoserver.org/), [Mapserver](https://mapserver.org/) or [QGIS Server](https://docs.qgis.org/3.16/en/docs/server_manual/index.html). Geoserver is arguably the most feature-rich software among those, however, unfortunately, it inherently lacks the ability to serve raster data from a PostGIS data base. 

# Custom WMS endpoint implementation
Since WMS itself is not that complex as a protocol, you can quite easily come up with your own WMS-compatible endpoint (at least one that suffices the `GetMap` capability).

## PL/pgSQL database routine
The first component is a quite comprehensive SQL function, that does most of the heavy lifting. Once created in your database it can be called as part of SQL queries with a couple of parameters. Connect to your PostGIS database and run the following script to initially create the function.

```sql
--- get_rast_tile.sql

create or replace function get_rast_tile(param_format text, param_width integer, param_height integer, param_srid integer, param_bbox text, param_schema text, param_table text) returns bytea
    immutable
    parallel safe
    cost 1000
    language plpgsql
as
$$
DECLARE
    var_sql text; var_result raster; var_srid integer;
    var_env geometry; var_env_buf geometry; var_erast raster;
BEGIN
    EXECUTE
        'SELECT ST_MakeEnvelope(' || array_to_string(('{' || param_bbox || '}')::float8[], ',') || ',' || param_srid ||')'
    INTO var_env;

    EXECUTE
        'SELECT ST_Buffer($1, 20);'
    USING var_env
    INTO var_env_buf; 

    var_sql :=
            'SELECT srid, ST_AsRaster($4,$5,$6,pixel_types,nodata_values,nodata_values) As erast
            FROM raster_columns
            WHERE r_table_schema = $1 AND r_table_name = $2 AND r_raster_column=$3';

    EXECUTE var_sql INTO var_srid, var_erast
        USING param_schema, param_table, 'rast', var_env, param_width, param_height;

    var_sql :=
        'WITH r AS (SELECT ST_Clip(rast,' ||
        CASE
            WHEN var_srid = param_srid THEN '$7'
            ELSE 'ST_Transform($7,$2)'
            END || ') As rast FROM  ' ||
        quote_ident(param_schema) || '.' ||
        quote_ident(param_table) || '
        WHERE ST_Intersects(rast,' ||
        CASE
            WHEN var_srid = param_srid THEN '$3'
            ELSE 'ST_Transform($3,$2)'
            END || ') limit 15)
        SELECT ST_Clip(ST_Union(rast), $3) As rast
        FROM (SELECT ST_Resample(' ||
        CASE
            WHEN var_srid = param_srid THEN 'rast'
            ELSE 'ST_Transform(rast,$1)'
            END ||
        ',$6,true,''NearestNeighbor'') As rast FROM r) As final';

    EXECUTE var_sql INTO var_result
        USING
            param_srid,
            var_srid,
            var_env,
            param_width,
            param_height,
            var_erast,
            var_env_buf;

    var_sql :=
        'SELECT ST_MapAlgebra($1, $2, ''[rast2]'', ''8BUI''::text, ''FIRST'', ''[rast2]'', NULL::text) rast';
    EXECUTE var_sql INTO var_result
    USING var_erast, var_result;

    IF var_result IS NULL THEN
        var_result := var_erast;
    END IF;

    RETURN
        CASE
            WHEN param_format ILIKE 'image/jpeg' THEN ST_AsJPEG(ST_ColorMap(var_result, 1, 'bluered'))
            ELSE ST_AsPNG(ST_ColorMap(var_result, 1, 'bluered'))
            END;
END;
$$;
```

The function takes a set of parameters, which mostly correspond to WMS parameters and it returns a fully rendered PNG (or JPEG) image as a byte array.

## Web endpoint
Second step is to implement the actual web endpoint, that can be called from GIS clients using HTTP. The following examples show implementations of a WMS endpoint (utilizing the above `get_rast_tile` SQL function) in two of the most popular Python web frameworks, [FastAPI](https://fastapi.tiangolo.com/) and [Flask](https://flask.palletsprojects.com/en/2.0.x/).

Both examples assume the presence of a PostGIS database called `geodb`, which includes a table `observations` for storing raster data. The table has to, at least, consist of a primary key column and a column `rast` of type `RASTER`. 

You can either create that table manually (not recommended), or use the command-line program [`raster2pgsql`](https://postgis.net/docs/using_raster_dataman.html) automatically create  an appropriate table structure.

### FastAPI
```python
#!/bin/python

# app.py

# Prerequisites:
# pip install fastapi, uvicorn, sqlalchemy[asyncpg], asyncpg

import typing
from io import BytesIO

import uvicorn as uvicorn
from fastapi import FastAPI, Query, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from starlette.responses import StreamingResponse

# Connect to database (asynchronously)
engine = create_async_engine('postgresql+asyncpg://user:password@localhost:5432/geodb', pool_pre_ping=True)
session = sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=AsyncSession)

# Initialize FastAPI app
app = FastAPI()


# Service method for retrieving WMS data from database
async def get_wms(db: AsyncSession, width: int, height: int, bbox: str, crs: int, format: str = 'image/png') -> typing.BinaryIO:
    async with db.begin():
        params = {
            'format': format,
            'width': width,
            'height': height,
            'crs': crs,
            'bbox': bbox,
            'table': 'observations'
        }

        # Required for ST_AsPNG and ST_AsJPEG to work
        await db.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';")
        await db.execute("SET postgis.enable_outdb_rasters TO true;")

        result = (await db.execute(text("SELECT get_rast_tile(:format, :width, :height, :crs, :bbox, 'public', :table)"), params)).first()
        return BytesIO(result[0])


def _get_db() -> typing.Generator[AsyncSession, None, None]:
    yield session()


# Web endpoint
@app.get("/wms")
async def wms(
        layers: str = Query(..., regex=r'^observations'),
        format: str = Query('image/png', regex='^image/(png|jpeg)$'),
        width: int = Query(..., gt=0),
        height: int = Query(..., gt=0),
        bbox: str = Query(..., regex=r'^[\d\.-]+,[\d\.-]+,[\d\.-]+,[\d\.-]+$'),
        crs: str = Query(..., regex=r'^EPSG:\d+$'),
        db: AsyncSession = Depends(_get_db)
):
    crs = int(crs.removeprefix('EPSG:'))
    image = await get_wms(db, width, height, bbox, crs, format)
    return StreamingResponse(image, media_type=format)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Flask
```python
#!/bin/python

# app.py

# Prerequisites:
# pip install flask, sqlalchemy[psycopg2], psycopg2-binary

import typing
from io import BytesIO

from flask import Flask, request, send_file
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Connect to database
engine = create_engine('postgresql://user:password@localhost:5432/geodb', pool_pre_ping=True)
session = sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False, class_=Session)

# Initialize Flask app
app = Flask(__name__)


# Service method for retrieving WMS data from database
def get_wms(db: Session, params: typing.Dict[str, typing.Any]) -> typing.BinaryIO:
    with db.begin():
        params['crs'] = int(params['crs'].removeprefix('EPSG:'))

        # Required for ST_AsPNG and ST_AsJPEG to work
        db.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';")
        db.execute("SET postgis.enable_outdb_rasters TO true;")

        result = db.execute(text("SELECT get_rast_tile(:format, :width, :height, :crs, :bbox, 'public', :layers)"), params).first()
        return BytesIO(result[0])


# Web endpoint
@app.route("/wms")
def wms():
    image = get_wms(session(), {**request.args})
    return send_file(image, mimetype=request.args['format'])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)

```

## Try it 
Use [`raster2pgsql`](https://postgis.net/docs/using_raster_dataman.html) to import raster data into your database in the first place. Afterwards, you can query it using WMS right from your browser. For instance, just try to go to:

```
http://localhost:8000/wms?
    width=256&
    height=256&
    bbox=49.007044,8.3946413,48.991105,8.413903&
    format=image/png&
    crs=EPSG:4326&
    layers=observations
```