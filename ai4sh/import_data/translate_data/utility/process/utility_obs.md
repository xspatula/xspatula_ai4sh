---
title:  "Utility input data"
layout: single
sidebar:
  nav: "utility"
excerpt: "Utility data are inserted to the database using standard xSpatula processes with arguments stated in json formatted process files. xSpatula also includes a process that converts spreadsheet data, either from csv files or excel, into properly formatted json process files. The easiest way to define utility data is thus to create a spreadsheet with the utility data and then run the process to convert the spreadsheet to json processs files following by running the json process file."
permalink: /utility/utility_obs
author_profile: false
date:   2026-02-17 16:13:03 +0200
last_modified_at:   2026-02-17 16:13:03 +0200
---

Utility data are inserted to the database using standard xSpatula processes with arguments stated in json formatted process files. xSpatula also includes a process that converts spreadsheet data, either from csv files or excel, into properly formatted json process files. The easiest way to define utility data is thus to create a spreadsheet with the utility data and then run the process to convert the spreadsheet to json processs files following by running the json process file.

## Processes

## General independent utilities

General stand alone utility tables (independent of other utility tables) for observation data can be filled beforehand and new data sequentially added.

### coordinate_system

Under construction

### apparatus

The observation_utility table apparatus defines all types of services, instruments and other equipment used for obtaining data.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| VISNIR spectrometer | visnir | Professional/citizen scientist Instrument ... |


Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_apparatus",
      "parameters": {
        "name": "VISNIR spectrometer",
        "alias": "visnir",
        "abstract": "Professional/citizen scientist Instrument that measures how light spanning the visible (400-750 nm) and near-infrared (750-2500 nm) regions, interacts with materials to determine their composition, chemical, or physical properties."
      }
    }
  ]
}
```

The online resource for xSpatula apparatus is available at [https://][#]

### quantity

The observation_utility table quantity defines all quantities for categorising observation values.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| Reflectance | reflectance | Measure of the proportion of light ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_quantity",
      "parameters": {
        "name": "Reflectance",
        "alias": "reflectance",
        "abstract": "Measure of the proportion of light or other radiation striking a surface which is reflected off it."
      }
    }
  ]
}
```

The online resource for xSpatula quantity is available at [https://][#]

### unit

The observation_utility table unit defines all units for recorded values entered in the database.

Spreadsheet (excel) data structure:

| name | alias | system | abstract |
| ----------- | ----------- | ----------- | ----------- |
| Kelvin | K | metric | Degrees Kelvin, temperature ratio scale ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_unit",
      "parameters": {
        "name": "Kelvin",
        "alias": "K",
        "system": "metric",
        "abstract": "Degrees Kelvin, temperature ratio scale starting at absolute zero"
      }
    }
  ]
}
```

The online resource for xSpatula unit is available at [https://][#]

### method

The observation_utility table method defines all analysis methods used for recorded values entered in the database. The methods entered must be precise as in many instances the same quantity and unit can be obtained with different methods that do not give compatible results.

Spreadsheet (excel) data structure:

| name | standard_schema | standard_code | deviation | url | abstract |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| cec iso 11260:1994 | iso | 11260:1994 | none | ... | Barium chloride (0.1 M) buffered ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_method",
      "parameters": {
        "name": "cec iso 11260:1994",
        "standardisation_schema": "iso",
        "standardisation_code": "11260:1994",
        "deviation": "none",
        "url": "none",
        "abstract": "Barium chloride (0.1 M) buffered solution, acid treatment, measurement of CO2 evolution, sample sieved<2mm base"
      }
    }
  ]
}
```

The online resource for xSpatula method is available at [https://][#]

### preservation

The observation_utility table preservation defines all preservation means used for analysis performed after long term samples storage. Long term can be as short as hours or days (for example for DNA/RNA analysis) or years. The storage conditions are recorded in a separate table.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| drying | dried | Samples dried after collection |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_preservation",
      "parameters": {
        "name": "drying",
        "alias": "dried",
        "abstract": "Samples dried after collection"
      }
    }
  ]
}
```

The online resource for xSpatula preservation is available at [https://][#]

### transport

The observation_utility table transport defines all transport means between sampling and analysis for analysis that are sensitive to changes under ambient conditions (for example enzymatic activity).

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| cold | refrigerator | Cold environment above 0 C |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_transport",
      "parameters": {
        "name": "cold",
        "alias": "refrigerator",
        "abstract": "Cold environment above 0 C"
      }
    }
  ]
}
```

The online resource for xSpatula transport is available at [https://][#]


### preparation

The observation_utility table preparation defines all preparation procedures applied before sample analysis, but after any preservation, storage and transport. Preparation procedures that are defined as part of a method are not required, but there is no harm in entering them either.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| air drying and sieving<2mm | air dried sieved<2mm | Air drying followed by ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_transport",
      "parameters": {
        "name": "air drying and sieving<2mm",
        "alias": "air dried sieved<2mm",
        "abstract": "Air drying followed by sieving to fraction < 2mm"
      }
    }
  ]
}
```

The online resource for xSpatula preparation is available at [https://][#]

### setting

The observation_utility table setting defines local setting systems relevant for samples of different classes, for example agriculture for soil samples taken at farms. The setting table is only a container, the actual relative setting positions belonging to a setting class are recorded in the table juxtaposition.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| agricultural fields | agriculture | Typical field locus (e.g. center, edge, plantline, alley) |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_setting",
      "parameters": {
        "name": "air drying and sieving<2mm",
        "alias": "air dried sieved<2mm",
        "abstract": "Air drying followed by sieving to fraction < 2mm"
      }
    }
  ]
}
```

The online resource for xSpatula setting is available at [https://][#]

### license

The observation_utility table license defines licences under which the datasets entered in the database fall.

Spreadsheet (excel) data structure:

| name | alias | url | doi |
| ----------- | ----------- | ----------- | ----------- |
| MIT license | MIT | https://opensource.org/license/mit | ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_license",
      "parameters": {
        "name": "MIT license",
        "alias": "MIT",
        "url": "https://opensource.org/license/mit"
      }
    }
  ]
}
```

The online resource for xSpatula license is available at [https://][#]

### order

The observation_utility table order (classification order) defines the most generalised class of a dataset, campaign or sample. For soil samples the order is thus soil. The classification can be broken down further into family, genus and species, but only the order is required for adding any dataset, campaign or sample.

Note, when entering a classification level definition (for example order) all lower levels in the classification hierarchy will automatically record the same class name. This both avoids the same name for different classes appearing at different hierarchical levels and also allows users to set the most general name at any hierarchical classification level.

Spreadsheet (excel) data structure:

| name | alias | abstract |
| ----------- | ----------- | ----------- |
| Soil | soil | Surface layer of the solid earth, typically ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_order",
      "parameters": {
        "name": "Soil",
        "alias": "soil",
        "abstract": "Surface layer of the solid earth, typically composed of organic remains and mineral particles."
      }
    }
  ]
}
```

The online resource for xSpatula order is available at [https://][#]

## General dependent utilities

Utility data with foreign keys (depending on records already existing in other utility tables) that can only be entered when the table holding the foreign contain the relevant record.

### Foreign keys

When entering records to tables that link to foreign keys (values in other tables) xSpatula uses a system where the name of the foreign key is stated as a text based name. When linking the foreign key xSpatula search for the name of the foreign key, but registered its automatically created id (always an integer number) in the new record. To achieve this, the name of the column with the the foreign key is always stated with a double underscore, where the first part (before the double underscore) denote the column name in the record to be written, and the second part indicate the table and column of the foreign key to search for (usually the name).

The next section (indicator) explains this with an example.

### indicator

The observation_utility table indicator defines specific properties reported by any service, instrument or other equipment.

Foreign key table: quantity

To indicate that an entered column represent a foreign key, the column header contains a double underscore (quantity_id__quantity_name). The first part of the column header (quantity_id) defines the table (quantity) and column (id) in the record to write (in the table indicator). The foreign key column always has a name combining the name of the foreign key table+_+column (in this case quantity_id). The second part of the  of the column header (quantity_name) is a hint to the user waht to enter and is then used by xSpatula to find the table and column to search for the foreign key in. In the database the foreign key will be registered as an id and not the name entered in the spreadsheet.

Spreadsheet (excel) data structure:

| name | alias | quantity_id__quantity_name |
| ----------- | ----------- | ----------- |
| pH-H2O | pH-water | pH |
| pH-CaCl2 | pH-calcium-chloride | pH |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_indicator",
      "parameters": {
        "name": "pH-H2O",
        "alias": "pH-water",
        "quantity_id__quantity_name": "pH"
      },
      {
        
        "process": "manage_indicator",
        "parameters": {
          "name": "pH-CaCl2",
          "alias": "pH-calcium-chloride",
          "quantity_id__quantity_name": "pH"
        }
      }
    }
  ]
}
```

In the example above, two different indicators (pH-H2O and pH-CaCl2) are recorded for the same quantity (pH). This is important as the difference in pH recorded with samples dissolved in water and calcium-chloride give different results, a difference that reveal more than a single pH observation.

### unit_translate

The observation_utility table unit_translate defines the translation from one unit to another. This is important as the units reported for the same indicator by different service, instrument or other equipment varies.

Foreign key table: unit

Spreadsheet (excel) data structure:

| src_unit_id__unit_name | dst_unit_id__unit_name | factor | addon | exponent | 
| ----------- | ----------- | ----------- | ----------- | ----------- |
| mm | m | 1000 | 0 | 1 |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_unit_translate",
      "parameters": {
        "src_unit_id__unit_name": "mm",
        "dst_unit_id__unit_name": "m",
        "factor": 1000,
        "addon": 0,
        "exponent": 1
      }
    }
  ]
}
```
### profiling

The observation_utility table profiling defines third dimension or z locus dimension system for samples requiring 3D registration.

Foreign key table: unit

Spreadsheet (excel) data structure:

| name | alias | unit_id__unit_name | abstract | 
| ----------- | ----------- | ----------- | ----------- |
| Depth in cm | depth_cm | cm | Depth in cm below reference surface |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_profiling",
      "parameters": {
        "name": "Depth in cm",
        "alias": "depth_cm",
        "unit_id__unit_name": "cm",
        "abstract": "Depth in cm below reference surface"
      }
    }
  ]
}
```

### juxtaposition

The observation_utility table juxtaposition defines qualitative neighbourhood locus from a local setting perspective. For example edge or center as juxtapositions vis-a-vis an agricultural field, where _agricultural field_ must then been apriori defined as a setting system in the table _setting_.

Foreign key table: setting

Spreadsheet (excel) data structure:

| setting_id__setting_name | name | alias | abstract | 
| ----------- | ----------- | ----------- | ----------- |
| agriculture	field | edge | edge | At field edge |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_juxtaposition",
      "parameters": {
        "setting_id__setting_name": "agriculture	field",
        "name": "field edge",
        "alias": "edge",
        "abstract": "At field edge"
      }
    }
  ]
}
```

### family

The observation_utility table family (classification family) defines an intermediate class of a dataset, campaign or sample. For soil samples the family could for instance be topsoil. The classification can be broken down further into genus and species, but only the order (parent of family) is required.

Foreign key table: order

Spreadsheet (excel) data structure:

| order_id__order_name | name | alias | abstract |
| ----------- | ----------- | ----------- | ----------- |
| soil | Topsoil | topsoil | The top layer of soil in which ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_family",
      "parameters": {
        "order_id__order_name": "soil",
        "name": "Topsoil",
        "alias": "topsoil",
        "abstract": "The top layer of soil in which typically vegetation grows"
      }
    }
  ]
}
```

### genus

The observation_utility table genus (classification genus) defines an intermediate class of a dataset, campaign or sample. For soil samples the genus could for instance be soil classes defined by FAO or USDA. The classification can be broken down further into  species, but only the order (the parent class of the entire classification hierarchy ) is required.

Foreign key table: family

Spreadsheet (excel) data structure:

| family_id__family_name | name | alias | abstract |
| ----------- | ----------- | ----------- | ----------- |
| topsoil | Andosol | andosol |  soil formed in volcanic materials |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_genus",
      "parameters": {
        "family_id__family_name": "topsoil",
        "name": "Andosol",
        "alias": "andosol",
        "abstract": "soil formed in volcanic materials"
      }
    }
  ]
}
```

### species

The observation_utility table species (classification species) defines the most detailed classification of a dataset, campaign or sample. The species is not required species, only the order (the parent class of the entire classification hierarchy) is required. In the classification hierarchy you can bypass any intermediate layer and always just state the order name, as in the example below.

Foreign key table: species

Spreadsheet (excel) data structure:

| genus_id__genus_name | name | alias | abstract |
| ----------- | ----------- | ----------- | ----------- |
| soil | behind barn | close to dung heap | nettle infested soil |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_species",
      "parameters": {
        "genus_id__genus_name": "soil",
        "name": "behind barn",
        "alias": "close to dung heap",
        "abstract": "nettle infested soil"
      }
    }
  ]
}
```

## Dataset specific utilities

Dataset specific utility tables include both independent tables with no foreign key and dependent tables with foreign keys.

### provider

The observation_utility table provider is somewhere between a general and a dataset specific table. It contains data on service providers, instruments and equipments that deliver analysis results. To accommodate different anaysis delivered by the same service provider (laboratory) to different datasets or campaigns, the provider table can not simply state the laboratory, but must state the laboratory in combination with the analysis performed for a particular dataset or campaign. This instruments it is assumed that each instrument (make and model) will generate the same results. If, however, results are delivered in for example different units (metric or imperial) then the also the instrument needs to be defined bound to the dataset or campaign receiving data with a deviating unit.

No foreign key dependency.

Spreadsheet (excel) data structure:

| name | full_name | alias | url | contact_name | contact_email |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| LUCAS-laboratory-2015 | LUCAS 2015 wet laboratory analysis | lucas-wetlab-2015 | https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data | European Soil Data center (ESDAC) |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_provider",
      "parameters": {
        "name": "LUCAS-laboratory-2015",
        "full_name": "LUCAS 2015 wet laboratory analysis",
        "alias": "lucas-wetlab-2015",
        "url": "https://esdac.jrc.ec.europa.eu/content/lucas2015-topsoil-data",
        "contact_name": " European Soil Data center (ESDAC)",
        "contact_email": "xy@z"
      }
    }
  ]
}
```

### provision

The observation_utility table provision defines the services, instruments and other equipments applied by the providers.

Foreign key tables: provider, apparatus

Spreadsheet (excel) data structure:

| provider_id__provider_name | apparatus_id__apparatus_name | name | alias | abstract |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| LUCAS-laboratory-2015 | wetlab | LUCAS-wetlab-2015 | lucas-soillab-2015 | Laboratory analysis package employed by ... |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_provider",
      "parameters": {
        "provider_id__provider_name": "LUCAS-laboratory-2015",
        "apparatus_id__apparatus_name": "wetlab",
        "name": "LUCAS-wetlab-2015",
        "alias": "lucas-soillab-2015",
        "abstract": "Laboratory analysis package employed by"
      }
    }
  ]
}
```

### provision_indicator

Foreign key tables: provision, indicator, method, unit

The observation_utility table provision_indicator defines the indicator (properties), the method applied and the unit reported for each and every observation derived from each and every provision.

Spreadsheet (excel) data structure:

| provision_id__provision_name | indicator_id__indicator_name | method_id__method_name | unit_id__unit_name |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| LUCAS-laboratory-2015 | CaCO3 | caco3 iso 10693:1994 | weight percent |

Json process command structure:

```
{
  "process": [
    {
      
      "process": "manage_provider",
      "parameters": {
        "provision_id__provision_name": "LUCAS-laboratory-2015",
        "indicator_id__indicator_name": "CaCO3",
        "method_id__method_name": "caco3 iso 10693:1994",
        "unit_id__unit_name": "weight percent"
      }
    }
  ]
}
```

## Acknowledgments and Funding

This work is part of the AI4SoilHealth project, funded by the European Union's Horizon Europe Research and Innovation Programme under Grant Agreement No. 101086179.

_Funded by the European Union. The views expressed are those of the authors and do not necessarily reflect those of the European Union or the European Research Executive Agency._

[github_aggregates_flat]: https://github.com/AI4SH/in-situ_data/tree/main/flat/aggregates

[github_aggregates_nested]: https://github.com/AI4SH/in-situ_data/tree/main/nested/aggregates

[slakes]: https://soilhealthinstitute.org/our-work/initiatives/slakes/
