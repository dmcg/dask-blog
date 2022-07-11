---
layout: post
title: Data Proximate Computation on a Dask Cluster Distributed Between Data Centres
author: Duncan McGregor (Met Office), Mike Grant (EUMETSAT), Richard Care (Met Office) 
tags: [distributed, deployment]
theme: twitter
---



_This work is a joint venture between the [Met Office](https://www.metoffice.gov.uk) and the [European Weather Cloud](https://www.europeanweather.cloud/), which is a partnership of [ECMWF](https://ecmwf.int) and [EUMETSAT](https://eumetsat.int/)._

## Summary

We have devised a technique for creating a Dask cluster where worker nodes are hosted in different data centres, connected by a mesh VPN that allows the scheduler and workers to communicate and exchange results. 

A novel (ab)use of Dask resources allows us to run data processing tasks on the workers in the cluster closest to the source data, so that communication between data centres is minimised. If combined with zarr to give access to huge hyper-cube datasets in object storage, we believe that the technique could realise the potential of data-proximate distributed computing in the Cloud.

## Introduction

The UK Met Office has been carrying out a study into data-proximate computing in collaboration with the European Weather Cloud. We identified Dask as a key technology, but whilst existing Dask techniques focus on parallel computation in a single data centre, we looked to extend this to computation across data centres. This way, when the data required is hosted in multiple locations, tasks can be run where the data is rather than copying it.

Dask worker nodes exchange data chunks over a network, coordinated by a scheduler. There is an assumption that all nodes are freely able to communicate, which is not generally true across data centres due to firewalls, NAT, etc, so a truly distributed approach has to solve this problem. In addition, it has to manage data transfer efficiently, because moving data chunks between data centres is much more costly than between workers in the same cloud.

This notebook documents a running proof-of-concept that addresses these problems. It runs a computation in 3 locations:

1. This computer, where the client and scheduler are running.  This was run on AWS during development.
2. The ECMWF data centre. This has compute resources, and hosts data containing *predictions*.
3. The EUMETSAT data centre, with compute resources and data on *observations*


```python
from IPython.display import Image
Image(filename="images/datacentres.png") # this because GitHub doesn't render markup images in private repos
```




    
![png](/images/dask-multi-cloud_1_0.png)
    



The idea is that tasks accessing data available in a location should be run there. Meanwhile the computation can be defined, invoked, and the results rendered, elsewhere. All this with minimal hinting to the computation as to how this should be done.

## Setup 

First some imports and conveniences


```python
import os
from time import sleep
import dask
from dask.distributed import Client
from dask.distributed import performance_report, get_task_stream
from dask_worker_pools import pool, propagate_pools
import pytest
import ipytest
import xarray
import matplotlib.pyplot as plt
from orgs import my_org
from tree import tree

ipytest.autoconfig()
```

In this case we are using a control plane IPv4 [WireGuard](https://www.wireguard.com/) network on 10.8.0.0/24 to set up the cluster - this is not a necessity, but simplifies this proof of concept. WireGuard peers are running on ECMWF and EUMETSAT machines already, but we have to start one here:


```python
!./start-wg.sh
```

    4: mo-aws-ec2: <POINTOPOINT,NOARP,UP,LOWER_UP> mtu 8921 qdisc noqueue state UNKNOWN group default qlen 1000
        link/none 
        inet 10.8.0.3/24 scope global mo-aws-ec2
           valid_lft forever preferred_lft forever


We have worker machines configured in both ECMWF and EUMETSAT, one in each. They are accessible on the control plane network as


```python
ecmwf_host='10.8.0.4'
%env ECMWF_HOST=$ecmwf_host
eumetsat_host='10.8.0.2'
%env EUMETSAT_HOST=$eumetsat_host
```

    env: ECMWF_HOST=10.8.0.4
    env: EUMETSAT_HOST=10.8.0.2


## Mount the Data

This machine needs access to the data files over the network in order to read NetCDF metadata. The workers are sharing their data with NFS, so we mount them here. (In this proof of concept, the control plane network is used for NFS, but the data plane network could equally be used, or a more appropriate technology such as [zarr](https://zarr.readthedocs.io/en/stable/) accessing object storage.)


```bash
%%bash
sudo ./data-reset.sh

mkdir -p /data/ecmwf
mkdir -p /data/eumetsat
sudo mount $ECMWF_HOST:/data/ecmwf /data/ecmwf
sudo mount $EUMETSAT_HOST:/eumetsatdata/ /data/eumetsat
```


```python
Image(filename="images/datacentres-data.png")
```




    
![png](/images/dask-multi-cloud_11_0.png)
    



## Access to Data

For this demonstration, we have two large data files that we want to process. On ECMWF we have predictions in `/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc`. Workers running in ECMWF can see the file


```python
!ssh -i ~/.ssh/id_rsa_rcar_infra  rcar@$ECMWF_HOST 'tree /data/'
```

    /data/
    └── ecmwf
        └── 000490262cdd067721a34112963bcaa2b44860ab.nc
    
    1 directory, 1 file


and because that directory is mounted here over NFS, so can this computer


```python
!tree /data/ecmwf
```

    /data/ecmwf
    └── 000490262cdd067721a34112963bcaa2b44860ab.nc
    
    0 directories, 1 file


This is a big file


```python
!ls -lh /data/ecmwf
```

    total 2.8G
    -rw-rw-r-- 1 ec2-user ec2-user 2.8G Mar 25 13:09 000490262cdd067721a34112963bcaa2b44860ab.nc


On EUMETSAT we have `observations.nc`


```python
!ssh -i ~/.ssh/id_rsa_rcar_infra  rcar@$EUMETSAT_HOST 'tree /data/eumetsat/ad-hoc'
```

    /data/eumetsat/ad-hoc
    └── observations.nc
    
    0 directories, 1 file


similarly visible on this computer


```python
!ls -lh /data/eumetsat/ad-hoc/observations.nc
```

    -rw-rw-r-- 1 613600004 613600004 4.8M May 20 10:57 /data/eumetsat/ad-hoc/observations.nc


Crucially, ECMWF data is not visible in the EUMETSAT data centre, and vice versa.

## Our Calculation

We want to compare the predictions against the observations.

We can open the predictions file with xarray


```python
predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')
predictions
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:                  (realization: 18, height: 33, latitude: 960,
                              longitude: 1280, bnds: 2)
Coordinates:
  * realization              (realization) int32 0 18 19 20 21 ... 31 32 33 34
  * height                   (height) float32 5.0 10.0 20.0 ... 5.5e+03 6e+03
  * latitude                 (latitude) float32 -89.91 -89.72 ... 89.72 89.91
  * longitude                (longitude) float32 -179.9 -179.6 ... 179.6 179.9
    forecast_period          timedelta64[ns] 1 days 18:00:00
    forecast_reference_time  datetime64[ns] 2021-11-07T06:00:00
    time                     datetime64[ns] 2021-11-09
Dimensions without coordinates: bnds
Data variables:
    air_pressure             (realization, height, latitude, longitude) float32 dask.array&lt;chunksize=(18, 33, 192, 160), meta=np.ndarray&gt;
    latitude_longitude       int32 -2147483647
    latitude_bnds            (latitude, bnds) float32 dask.array&lt;chunksize=(960, 2), meta=np.ndarray&gt;
    longitude_bnds           (longitude, bnds) float32 dask.array&lt;chunksize=(1280, 2), meta=np.ndarray&gt;
Attributes:
    history:                      2021-11-07T10:27:38Z: StaGE Decoupler
    institution:                  Met Office
    least_significant_digit:      1
    mosg__forecast_run_duration:  PT198H
    mosg__grid_domain:            global
    mosg__grid_type:              standard
    mosg__grid_version:           1.6.0
    mosg__model_configuration:    gl_ens
    source:                       Met Office Unified Model
    title:                        MOGREPS-G Model Forecast on Global 20 km St...
    um_version:                   11.5
    Conventions:                  CF-1.7</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-7315ddc5-ed0c-48c3-87a5-b85f32410f6c' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-7315ddc5-ed0c-48c3-87a5-b85f32410f6c' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>realization</span>: 18</li><li><span class='xr-has-index'>height</span>: 33</li><li><span class='xr-has-index'>latitude</span>: 960</li><li><span class='xr-has-index'>longitude</span>: 1280</li><li><span>bnds</span>: 2</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-be4be477-6c6e-4c9c-a57e-ed94d9b78361' class='xr-section-summary-in' type='checkbox'  checked><label for='section-be4be477-6c6e-4c9c-a57e-ed94d9b78361' class='xr-section-summary' >Coordinates: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>realization</span></div><div class='xr-var-dims'>(realization)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 18 19 20 21 22 ... 30 31 32 33 34</div><input id='attrs-62e5bd32-1b6f-4582-b8d7-7d4924766c0a' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-62e5bd32-1b6f-4582-b8d7-7d4924766c0a' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-4f4d118d-9ec4-4c49-9c30-aaf460a347e1' class='xr-var-data-in' type='checkbox'><label for='data-4f4d118d-9ec4-4c49-9c30-aaf460a347e1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>units :</span></dt><dd>1</dd><dt><span>standard_name :</span></dt><dd>realization</dd></dl></div><div class='xr-var-data'><pre>array([ 0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
      dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>height</span></div><div class='xr-var-dims'>(height)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>5.0 10.0 20.0 ... 5.5e+03 6e+03</div><input id='attrs-65f18df2-ba93-447b-a2de-65c115f116da' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-65f18df2-ba93-447b-a2de-65c115f116da' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fcb85a4c-0368-4945-be6f-f5c04c9220dc' class='xr-var-data-in' type='checkbox'><label for='data-fcb85a4c-0368-4945-be6f-f5c04c9220dc' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>axis :</span></dt><dd>Z</dd><dt><span>units :</span></dt><dd>m</dd><dt><span>standard_name :</span></dt><dd>height</dd><dt><span>positive :</span></dt><dd>up</dd></dl></div><div class='xr-var-data'><pre>array([5.00e+00, 1.00e+01, 2.00e+01, 3.00e+01, 5.00e+01, 7.50e+01, 1.00e+02,
       1.50e+02, 2.00e+02, 2.50e+02, 3.00e+02, 4.00e+02, 5.00e+02, 6.00e+02,
       7.00e+02, 8.00e+02, 1.00e+03, 1.25e+03, 1.50e+03, 1.75e+03, 2.00e+03,
       2.25e+03, 2.50e+03, 2.75e+03, 3.00e+03, 3.25e+03, 3.50e+03, 3.75e+03,
       4.00e+03, 4.50e+03, 5.00e+03, 5.50e+03, 6.00e+03], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>latitude</span></div><div class='xr-var-dims'>(latitude)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-89.91 -89.72 ... 89.72 89.91</div><input id='attrs-d885186f-bb1e-41d6-8e54-d6614acdb3a1' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-d885186f-bb1e-41d6-8e54-d6614acdb3a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-bf998e85-24cd-4b87-a15e-74db8feaed0e' class='xr-var-data-in' type='checkbox'><label for='data-bf998e85-24cd-4b87-a15e-74db8feaed0e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>axis :</span></dt><dd>Y</dd><dt><span>bounds :</span></dt><dd>latitude_bnds</dd><dt><span>units :</span></dt><dd>degrees_north</dd><dt><span>standard_name :</span></dt><dd>latitude</dd></dl></div><div class='xr-var-data'><pre>array([-89.90625, -89.71875, -89.53125, ...,  89.53125,  89.71875,  89.90625],
      dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>longitude</span></div><div class='xr-var-dims'>(longitude)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>-179.9 -179.6 ... 179.6 179.9</div><input id='attrs-a2d47e60-ef3a-4363-aba5-28dc1a87d75c' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-a2d47e60-ef3a-4363-aba5-28dc1a87d75c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b286b9e4-2858-46dd-bfea-ec9735bd247f' class='xr-var-data-in' type='checkbox'><label for='data-b286b9e4-2858-46dd-bfea-ec9735bd247f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>axis :</span></dt><dd>X</dd><dt><span>bounds :</span></dt><dd>longitude_bnds</dd><dt><span>units :</span></dt><dd>degrees_east</dd><dt><span>standard_name :</span></dt><dd>longitude</dd></dl></div><div class='xr-var-data'><pre>array([-179.85938, -179.57812, -179.29688, ...,  179.29688,  179.57812,
        179.85938], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>forecast_period</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-f5115bfa-789d-4bf7-b6df-44c0cce0fccb' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-f5115bfa-789d-4bf7-b6df-44c0cce0fccb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-79fe7672-5aba-42d4-b57e-b7866df9325a' class='xr-var-data-in' type='checkbox'><label for='data-79fe7672-5aba-42d4-b57e-b7866df9325a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>standard_name :</span></dt><dd>forecast_period</dd></dl></div><div class='xr-var-data'><pre>array(151200000000000, dtype=&#x27;timedelta64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>forecast_reference_time</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-e16ffdcb-d304-4647-af5d-e18042371265' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-e16ffdcb-d304-4647-af5d-e18042371265' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-a8ef5e26-a677-48af-ae7f-fd0133791eba' class='xr-var-data-in' type='checkbox'><label for='data-a8ef5e26-a677-48af-ae7f-fd0133791eba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>standard_name :</span></dt><dd>forecast_reference_time</dd></dl></div><div class='xr-var-data'><pre>array(&#x27;2021-11-07T06:00:00.000000000&#x27;, dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>time</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-be739ef0-fbf8-4544-96ae-8b133c93e572' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-be739ef0-fbf8-4544-96ae-8b133c93e572' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f3657616-5d98-4d6c-be61-40508161c4de' class='xr-var-data-in' type='checkbox'><label for='data-f3657616-5d98-4d6c-be61-40508161c4de' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>standard_name :</span></dt><dd>time</dd></dl></div><div class='xr-var-data'><pre>array(&#x27;2021-11-09T00:00:00.000000000&#x27;, dtype=&#x27;datetime64[ns]&#x27;)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-8d0c8485-e563-4e89-8115-a684a5bb27c9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-8d0c8485-e563-4e89-8115-a684a5bb27c9' class='xr-section-summary' >Data variables: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>air_pressure</span></div><div class='xr-var-dims'>(realization, height, latitude, longitude)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(18, 33, 192, 160), meta=np.ndarray&gt;</div><input id='attrs-c69f4511-6a4f-4719-87c2-b1d59f4d44b3' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-c69f4511-6a4f-4719-87c2-b1d59f4d44b3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f8977552-0a98-4fe8-8229-3f347e293840' class='xr-var-data-in' type='checkbox'><label for='data-f8977552-0a98-4fe8-8229-3f347e293840' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>standard_name :</span></dt><dd>air_pressure</dd><dt><span>units :</span></dt><dd>Pa</dd><dt><span>grid_mapping :</span></dt><dd>latitude_longitude</dd></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 2.72 GiB </td>
                        <td> 69.61 MiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (18, 33, 960, 1280) </td>
                        <td> (18, 33, 192, 160) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 41 Tasks </td>
                        <td> 40 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float32 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="381" height="157" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="27" y2="0" style="stroke-width:2" />
  <line x1="0" y1="25" x2="27" y2="25" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="25" style="stroke-width:2" />
  <line x1="27" y1="0" x2="27" y2="25" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 27.118768537103147,0.0 27.118768537103147,25.412616514582485 0.0,25.412616514582485" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="13.559384" y="45.412617" font-size="1.0rem" font-weight="100" text-anchor="middle" >18</text>
  <text x="47.118769" y="12.706308" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(0,47.118769,12.706308)">1</text>


  <!-- Horizontal lines -->
  <line x1="97" y1="0" x2="114" y2="17" style="stroke-width:2" />
  <line x1="97" y1="18" x2="114" y2="35" />
  <line x1="97" y1="36" x2="114" y2="53" />
  <line x1="97" y1="54" x2="114" y2="71" />
  <line x1="97" y1="72" x2="114" y2="89" />
  <line x1="97" y1="90" x2="114" y2="107" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="97" y1="0" x2="97" y2="90" style="stroke-width:2" />
  <line x1="114" y1="17" x2="114" y2="107" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="97.0,0.0 114.93626877434578,17.93626877434578 114.93626877434578,107.93626877434578 97.0,90.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="97" y1="0" x2="217" y2="0" style="stroke-width:2" />
  <line x1="114" y1="17" x2="234" y2="17" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="97" y1="0" x2="114" y2="17" style="stroke-width:2" />
  <line x1="112" y1="0" x2="129" y2="17" />
  <line x1="127" y1="0" x2="144" y2="17" />
  <line x1="142" y1="0" x2="159" y2="17" />
  <line x1="157" y1="0" x2="174" y2="17" />
  <line x1="172" y1="0" x2="189" y2="17" />
  <line x1="187" y1="0" x2="204" y2="17" />
  <line x1="202" y1="0" x2="219" y2="17" />
  <line x1="217" y1="0" x2="234" y2="17" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="97.0,0.0 217.0,0.0 234.9362687743458,17.93626877434578 114.93626877434578,17.93626877434578" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="114" y1="17" x2="234" y2="17" style="stroke-width:2" />
  <line x1="114" y1="35" x2="234" y2="35" />
  <line x1="114" y1="53" x2="234" y2="53" />
  <line x1="114" y1="71" x2="234" y2="71" />
  <line x1="114" y1="89" x2="234" y2="89" />
  <line x1="114" y1="107" x2="234" y2="107" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="114" y1="17" x2="114" y2="107" style="stroke-width:2" />
  <line x1="129" y1="17" x2="129" y2="107" />
  <line x1="144" y1="17" x2="144" y2="107" />
  <line x1="159" y1="17" x2="159" y2="107" />
  <line x1="174" y1="17" x2="174" y2="107" />
  <line x1="189" y1="17" x2="189" y2="107" />
  <line x1="204" y1="17" x2="204" y2="107" />
  <line x1="219" y1="17" x2="219" y2="107" />
  <line x1="234" y1="17" x2="234" y2="107" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="114.93626877434578,17.93626877434578 234.93626877434576,17.93626877434578 234.93626877434576,107.93626877434578 114.93626877434578,107.93626877434578" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="174.936269" y="127.936269" font-size="1.0rem" font-weight="100" text-anchor="middle" >1280</text>
  <text x="254.936269" y="62.936269" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,254.936269,62.936269)">960</text>
  <text x="95.968134" y="118.968134" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,95.968134,118.968134)">33</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>latitude_longitude</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>...</div><input id='attrs-3f87c973-a583-46d7-93ea-7875bafb4da2' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-3f87c973-a583-46d7-93ea-7875bafb4da2' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1aee046f-bfe0-4d73-bf78-2aaef489788d' class='xr-var-data-in' type='checkbox'><label for='data-1aee046f-bfe0-4d73-bf78-2aaef489788d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>grid_mapping_name :</span></dt><dd>latitude_longitude</dd><dt><span>longitude_of_prime_meridian :</span></dt><dd>0.0</dd><dt><span>earth_radius :</span></dt><dd>6371229.0</dd></dl></div><div class='xr-var-data'><pre>array(-2147483647, dtype=int32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>latitude_bnds</span></div><div class='xr-var-dims'>(latitude, bnds)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(960, 2), meta=np.ndarray&gt;</div><input id='attrs-16076922-e017-4a45-8f4c-bfe9426a1a62' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-16076922-e017-4a45-8f4c-bfe9426a1a62' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d075b5f1-6f4e-4f03-8fa8-e45b047b7827' class='xr-var-data-in' type='checkbox'><label for='data-d075b5f1-6f4e-4f03-8fa8-e45b047b7827' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 7.50 kiB </td>
                        <td> 7.50 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (960, 2) </td>
                        <td> (960, 2) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 2 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float32 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="75" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="25" y2="0" style="stroke-width:2" />
  <line x1="0" y1="120" x2="25" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="25" y1="0" x2="25" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 25.412616514582485,0.0 25.412616514582485,120.0 0.0,120.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="12.706308" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >2</text>
  <text x="45.412617" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,45.412617,60.000000)">960</text>
</svg>
        </td>
    </tr>
</table></div></li><li class='xr-var-item'><div class='xr-var-name'><span>longitude_bnds</span></div><div class='xr-var-dims'>(longitude, bnds)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>dask.array&lt;chunksize=(1280, 2), meta=np.ndarray&gt;</div><input id='attrs-65520dd7-ceef-4ef7-ad82-04c57ff95da3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-65520dd7-ceef-4ef7-ad82-04c57ff95da3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-46dea395-0552-458e-bc3f-8cf3289266ee' class='xr-var-data-in' type='checkbox'><label for='data-46dea395-0552-458e-bc3f-8cf3289266ee' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 10.00 kiB </td>
                        <td> 10.00 kiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (1280, 2) </td>
                        <td> (1280, 2) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 2 Tasks </td>
                        <td> 1 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float32 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="75" height="170" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="0" y1="0" x2="25" y2="0" style="stroke-width:2" />
  <line x1="0" y1="120" x2="25" y2="120" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="0" y1="0" x2="0" y2="120" style="stroke-width:2" />
  <line x1="25" y1="0" x2="25" y2="120" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="0.0,0.0 25.412616514582485,0.0 25.412616514582485,120.0 0.0,120.0" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Text -->
  <text x="12.706308" y="140.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" >2</text>
  <text x="45.412617" y="60.000000" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,45.412617,60.000000)">1280</text>
</svg>
        </td>
    </tr>
</table></div></li></ul></div></li><li class='xr-section-item'><input id='section-64719674-8f31-4c88-85a1-bcff12e3ba08' class='xr-section-summary-in' type='checkbox'  ><label for='section-64719674-8f31-4c88-85a1-bcff12e3ba08' class='xr-section-summary' >Attributes: <span>(12)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>history :</span></dt><dd>2021-11-07T10:27:38Z: StaGE Decoupler</dd><dt><span>institution :</span></dt><dd>Met Office</dd><dt><span>least_significant_digit :</span></dt><dd>1</dd><dt><span>mosg__forecast_run_duration :</span></dt><dd>PT198H</dd><dt><span>mosg__grid_domain :</span></dt><dd>global</dd><dt><span>mosg__grid_type :</span></dt><dd>standard</dd><dt><span>mosg__grid_version :</span></dt><dd>1.6.0</dd><dt><span>mosg__model_configuration :</span></dt><dd>gl_ens</dd><dt><span>source :</span></dt><dd>Met Office Unified Model</dd><dt><span>title :</span></dt><dd>MOGREPS-G Model Forecast on Global 20 km Standard Grid</dd><dt><span>um_version :</span></dt><dd>11.5</dd><dt><span>Conventions :</span></dt><dd>CF-1.7</dd></dl></div></li></ul></div></div>



Dask code running on this machine has read the metadata for the file via NFS, but has not yet read in the data arrays themselves. 

Likewise we can see the observations, so we can perform a calculation locally. Here we average the predictions over the realisations and then compare them with the observations at a particular height. (This is a deliberately inefficient calculation, as we could average at only the required height, but you get the point.)


```python
%%time
def scope():
    client = Client()
    predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')
    observations = xarray.open_dataset('/data/eumetsat/ad-hoc/observations.nc').chunk('auto')    

    averages = predictions.mean('realization')
    diff = averages.isel(height=10) - observations
    diff.compute()
    
#scope()    
```

    CPU times: user 10 µs, sys: 2 µs, total: 12 µs
    Wall time: 13.8 µs


When we uncomment `scope()` and actually run this, it takes over 14 minutes to complete! Accessing the data over NFS between data centres (we run this notebook in  AWS) is just too slow.

In fact just copying the data files onto the computer running this notebook takes the same sort of time. At least 2.8 GiB + 4.8 MiB of data must pass from the data centres to this machine to perform the calculation.

Instead we should obviously run the Dask tasks where the data is. We can do that on a Dask cluster.

## Running Up a Cluster

The cluster is run up with a single command. It takes a while though


```python
import subprocess

scheduler_process = subprocess.Popen([
        '../dask_multicloud/dask-boot.sh', 
        f"rcar@{ecmwf_host}",
        f"rcar@{eumetsat_host}"
    ])
```

    [#] ip link add dasklocal type wireguard
    [#] wg setconf dasklocal /dev/fd/63
    [#] ip -6 address add fda5:c0ff:eeee:0::1/64 dev dasklocal
    [#] ip link set mtu 1420 up dev dasklocal
    [#] ip -6 route add fda5:c0ff:eeee:2::/64 dev dasklocal
    [#] ip -6 route add fda5:c0ff:eeee:1::/64 dev dasklocal
    2022-06-29 14:46:57,237 - distributed.scheduler - INFO - -----------------------------------------------
    2022-06-29 14:46:58,602 - distributed.http.proxy - INFO - To route to workers diagnostics web server please install jupyter-server-proxy: python -m pip install jupyter-server-proxy
    2022-06-29 14:46:58,643 - distributed.scheduler - INFO - -----------------------------------------------
    2022-06-29 14:46:58,644 - distributed.scheduler - INFO - Clear task state
    2022-06-29 14:46:58,646 - distributed.scheduler - INFO -   Scheduler at:     tcp://172.17.0.2:8786
    2022-06-29 14:46:58,646 - distributed.scheduler - INFO -   dashboard at:                     :8787
    2022-06-29 14:47:16,104 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:1::11]:37977', name: ecmwf-1-2, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:16,107 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:1::11]:37977
    2022-06-29 14:47:16,108 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:16,108 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:1::11]:44575', name: ecmwf-1-3, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:16,109 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:1::11]:44575
    2022-06-29 14:47:16,109 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:16,113 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:1::11]:40121', name: ecmwf-1-1, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:16,114 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:1::11]:40121
    2022-06-29 14:47:16,114 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:16,119 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:1::11]:40989', name: ecmwf-1-0, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:16,121 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:1::11]:40989
    2022-06-29 14:47:16,121 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:23,342 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:2::11]:33423', name: eumetsat-2-0, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:23,343 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:2::11]:33423
    2022-06-29 14:47:23,343 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:23,346 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:2::11]:43953', name: eumetsat-2-1, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:23,348 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:2::11]:43953
    2022-06-29 14:47:23,348 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:23,350 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:2::11]:46089', name: eumetsat-2-3, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:23,352 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:2::11]:46089
    2022-06-29 14:47:23,352 - distributed.core - INFO - Starting established connection
    2022-06-29 14:47:23,357 - distributed.scheduler - INFO - Register worker <WorkerState 'tcp://[fda5:c0ff:eeee:2::11]:43727', name: eumetsat-2-2, status: undefined, memory: 0, processing: 0>
    2022-06-29 14:47:23,358 - distributed.scheduler - INFO - Starting worker compute stream, tcp://[fda5:c0ff:eeee:2::11]:43727
    2022-06-29 14:47:23,358 - distributed.core - INFO - Starting established connection


We need to wait for 8 `distributed.core - INFO - Starting established connection` lines - one from each of 4 worker processes on each of 2 worker machines.

What has happened here is:

1. `start-scheduler.sh` runs up a Docker container on this computer.
2. The container creates a WireGuard IPv6 data plane VPN. This involves generating shared keys for all the nodes and a network interface inside itself. This data plane VPN is transient and unique to this cluster.
3. The container runs a Dask scheduler, hosted on the data plane network.
4. It then asks each data centre to provision workers and routing.

Each data centre hosts a control process, accessible over the control plane network. On invocation:

1. The control process creates a WireGuard network interface on the data plane network. This acts as a router between the workers inside the data centres and the scheduler.
2. It starts Docker containers on compute instances. These containers have their own WireGuard network interface on the data plane network, routing via the control process instance.
3. The Docker containers spawn (4) Dask worker processes, each of which connects via the data plane network back to the scheduler created at the beginning.

The result is one container on this computer running the scheduler, talking to a container on each worker machine, over a throw-away data plane WireGuard IPv6 network which allows each of the (in this case 8) Dask worker processes to communicate with each other and the scheduler, even though they are partitioned over 3 data centres.

Something like this


```python
Image(filename="images/datacentres-dask.png")
```




    
![png](/images/dask-multi-cloud_35_0.png)
    



Key

* <span style='color: blue'>Data plane network</span>
* <span style='color: green'>Dask</span>
* <span style='color: red'>NetCDF data</span>

## Connecting to the Cluster

The scheduler for the cluster is now running in a Docker container on this machine and is exposed on `localhost`, so we can create a client talking to it


```python
client = Client("localhost:8786")
```

    2022-06-29 14:47:35,535 - distributed.scheduler - INFO - Receive client connection: Client-69f22f41-f7ba-11ec-a0a2-0acd18a5c05a
    2022-06-29 14:47:35,536 - distributed.core - INFO - Starting established connection
    /home/ec2-user/miniconda3/envs/jupyter/lib/python3.10/site-packages/distributed/client.py:1287: VersionMismatchWarning: Mismatched versions found
    
    +---------+--------+-----------+---------+
    | Package | client | scheduler | workers |
    +---------+--------+-----------+---------+
    | msgpack | 1.0.4  | 1.0.3     | 1.0.3   |
    | numpy   | 1.23.0 | 1.22.3    | 1.22.3  |
    | pandas  | 1.4.3  | 1.4.2     | 1.4.2   |
    +---------+--------+-----------+---------+
    Notes: 
    -  msgpack: Variation is ok, as long as everything is above 0.6
      warnings.warn(version_module.VersionMismatchWarning(msg[0]["warning"]))


If you click through the client you should see the workers under the `Scheduler Info` node


```python
# client
```

You can also click through to the Dashboard on http://localhost:8787/status. There we can show the workers on the task stream


```python
def show_all_workers():
    my_org().compute(workers='ecmwf-1-0')
    my_org().compute(workers='ecmwf-1-1')
    my_org().compute(workers='ecmwf-1-2')
    my_org().compute(workers='ecmwf-1-3')
    my_org().compute(workers='eumetsat-2-0')
    my_org().compute(workers='eumetsat-2-1')
    my_org().compute(workers='eumetsat-2-2')
    my_org().compute(workers='eumetsat-2-3')
    sleep(0.5)

show_all_workers()
```

## Running on the Cluster

Now that there is a Dask client in scope, calculations will be run on the cluster. We can define the tasks to be run


```python
predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')
observations = xarray.open_dataset('/data/eumetsat/ad-hoc/observations.nc').chunk('auto')    
    
averages = predictions.mean('realization')
diff = averages.isel(height=10) - observations
```

But when we try to perform the calculation it fails


```python
with pytest.raises(FileNotFoundError) as excinfo:
    show_all_workers()
    diff.compute()    

str(excinfo.value)
```




    "[Errno 2] No such file or directory: b'/data/eumetsat/ad-hoc/observations.nc'"



It fails because the Dask scheduler has sent some of the tasks to read the data to workers running in EUMETSAT. They cannot see the data in ECMWF, and nor do we want them too, because reading all that data between data centres would be too slow.

# Data-Proximate Computation

Dask has the concept of [resources](https://distributed.dask.org/en/stable/resources.html). Tasks can be scheduled to run only where a resource (such as a GPU or amount of RAM) is available. We can [abuse this mechanism](https://dask.discourse.group/t/understanding-work-stealing/335/13) to pin tasks to a data centre, by treating the data centre as a resource.

To do this, when we create the workers we mark them as having a `pool-ecmwf` or `pool-eumetsat` resource. Then when we want to create tasks that can only run in one data centre, we annotate them as requiring the appropriate resource


```python
with (dask.annotate(resources={'pool-ecmwf': 1})):
    predictions.mean('realization').isel(height=10).compute()
```

We can hide that boilerplate inside a Python context manager - `pool` - and write


```python
with pool('ecmwf'):    
    predictions.mean('realization').isel(height=10).compute()
```

The `pool` context manager is a collaboration with the Dask developers, and is [published on GitHub](https://github.com/gjoseph92/dask-worker-pools). You can read more on the evolution of the concept on the [Dask Discourse](https://dask.discourse.group/t/understanding-work-stealing/335).

We can do better than annotating the computation tasks though. If we load the data inside the context manager block, the data loading tasks will carry the annotation with them


```python
with pool('ecmwf'):
    predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')
```

In this case we need another context manager in the library, `propagate_pools`, to ensure that the annotation is not lost when the task graph is processed and executed


```python
with propagate_pools():
    predictions.mean('realization').isel(height=10).compute()
```

The two context managers allow us to annotate data with its pool, and hence where the loading tasks will run


```python
with pool('ecmwf'):    
    predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')

with pool('eumetsat'):
    observations = xarray.open_dataset('/data/eumetsat/ad-hoc/observations.nc').chunk('auto')
```

define some deferred calculations oblivious to the data's provenance


```python
averaged_predictions = predictions.mean('realization')
diff = averaged_predictions.isel(height=10) - observations
```

and then perform the final calculation


```python
%%time 
with propagate_pools():
    show_all_workers()
    diff.compute()
```

    CPU times: user 127 ms, sys: 6.34 ms, total: 133 ms
    Wall time: 4.88 s


Remember, our aim was to distribute a calculation across data centres, whilst preventing workers reading foreign bulk data.

Here we know that data is only being read by workers in the appropriate location, because neither data centre can read the other's data. Once data is in memory, Dask prefers to schedule tasks on the workers that have it, so that the local workers will tend to perform follow-on calcuations, and data chunks will tend to stay in the data centre that they were read from.

Ordinarily though, if workers are idle, Dask would use them to perform calculations even if they don't have the data. If allowed, this work stealing would result in data being moved between data centres unnecessarly, a potentially expensive operation. To prevent this, the `propagate_pools` context manager installs a scheduler optimisation that disallows work-stealing between workers in different pools.

Once data loaded in one pool needs to be combined with data from another (the substraction in `averaged_predictions.isel(height=10) - observations` above), this is no longer classified as work stealing, and Dask will move data between data centres as required.

That calculation in one go looks like this


```python
%%time
with pool('ecmwf'):    
    predictions = xarray.open_dataset('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc').chunk('auto')

with pool('eumetsat'):
    observations = xarray.open_dataset('/data/eumetsat/ad-hoc/observations.nc').chunk('auto')   

averages = predictions.mean('realization')
diff = averages.isel(height=10) - observations
    
with propagate_pools():
    show_all_workers()
    plt.figure(figsize=(6, 6))
    plt.imshow(diff.to_array()[0,...,0], origin='lower')
```

    CPU times: user 234 ms, sys: 27.6 ms, total: 261 ms
    Wall time: 6.04 s



    
![png](/images/dask-multi-cloud_65_1.png)
    


In terms of code, compared with the local version above, this has only added the use of `with` blocks to label data and manage execution, and executes some 100 times faster.

This is a best case, because in the demonstrator the files are actually hosted on the worker machines, so the speed difference between reading the files locally and reading them over NFS is maximized. Perhaps more persuasive is the measured volume of network traffic.

| Method | Time Taken | Measured Network Traffic|
| :-- | :-: | :-: |
| Calculation over NFS | > 14 minutes | 2.8 GiB |
| Distributed Calculation | ~ 10 seconds | 8 MiB |

Apart from some control and status messages, only the data required to paint the picture is sent over the network to this computer.

Looking at the task stream we see the ECMWF workers (on the bottom) doing the bulk of the reading and computation, with the red transfer tasks joining this with data on EUMETSAT.


```python
Image(filename="images/task-graph.png")
```




    
![png](/images/dask-multi-cloud_67_0.png)
    



## Catalogs

We can simplify this code even more. Because the data-loading tasks are labelled with their resource pool, this can be opaque to the scientist. So we can write


```python
def load_from_catalog(path):
    with pool(path.split('/')[2]):
        return xarray.open_dataset(path).chunk('auto')
```

allowing us to ignore where the data came from


```python
predictions = load_from_catalog('/data/ecmwf/000490262cdd067721a34112963bcaa2b44860ab.nc')
observations = load_from_catalog('/data/eumetsat/ad-hoc/observations.nc')  

averages = predictions.mean('realization')
diff = averages.isel(height=10) - observations

with propagate_pools():
    show_all_workers()
    diff.compute()
```

Of course the cluster would have to be provisioned with compute resources in the appropriate data centres, although with some work this could be made dynamic as part of the catalog code.


## More Information

This [notebook](https://github.com/dmcg/dask-multicloud-poc/blob/main/demo/dask-multi-cloud.ipynb), and the code behind it, are published in a [GitHub repository](https://github.com/dmcg/dask-multicloud-poc). 

For details of the prototype implementation, and ideas for enhancements, see
[dask-multi-cloud-details.ipynb](https://github.com/dmcg/dask-multicloud-poc/blob/main/demo/dask-multi-cloud-details.ipynb).

## Acknowledgements

Thank you to Armagan Karatosun (EUMETSAT) and Vasileios Baousis (ECMWF) for their help and support with the infrastructure to support this proof of concept.
Gabe Joseph (Coiled) wrote the clever pool context managers, and Jacob Tomlinson (NVIDIA) reviewed this document.

_Crown Copyright 2022_


```python

```
