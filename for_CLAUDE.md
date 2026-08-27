## Single step import from excel or csv data

I want to create an additional process for single step for data INSERT from excel or csv data 


## objective
Creaete a signle step INSERT process from excel or csv data - while also keeping the 2 step process of keeping the translation of excel to JSON and then INSER/UPDATE from the JSON. 

## outline

At present data INSERT/UPDATE is via 2-step process of first translating data from excel or csv to JSON files - and then read the JSON for the actual database processing. I still need this route, both because some data are only available as JSON object files, for being able to read and control the JSON files manually when errors occur and for keeping the UPDATE possibility. But for ordinary users it would be more straight forward to only use a single notebook cell/step for INSERT. I thus want an additional process - for direct INSERT (no UPDATE option) from excel or csv data.

I want ot keep the existing notebooks with the 2-step process and create new ones for the single step insert. The existing notebooks are under @ai4sh/import_data.