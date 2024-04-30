# pyCDI 

Python implementation of the producing the composite drought indicator (CDI) based on openly available STAC imagery

The CDI incorporates three main components that impact drought severity: 1) precipitation deficit, 2) excess temperature and 3) vegetation response, incorporating within the CDI three drought indices:

The CDI incorporates **three main components** that impact drought severity: 1) precipitation deficit, 2) excess temperature and 3) vegetation response, incorporating within the CDI three drought indices:

- Precipiation Drought Index (PDI)
- Temperature Drought Index (TDI)
- Vegetation Drought Index (VDI)

A jupyter notebook is provided to explain how each of the drought indicators are processed and eventually merged to produce CDI

# Instalation 

The following libraries will be needed to properly use the Jupyter Notebook:

- numpy 
- pandas
- geopandas
- rasterio
- xarray
- rioxarray
- cubo[ee] # this install cubo and also all GEE dependencies
- leafmap
- notebook 

Functions were tested with python=3.9

# Contributors
Vicente Burchard-Levine vburchard@ica.csic.es

Hector Nieto 

Tinebeb Yohannes

Elias Cherenet Weldemariam

Getachew Mehabie Mulualem

Ana Andreu

# License  
pyCDI: a Python implementation of the composite drought index (CDI)

Copyright 2024 Vicente Burchard-Levine and contributors.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
