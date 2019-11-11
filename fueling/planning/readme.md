## Stability Reference Table 

The table contains two sub-tables: **lat_jerk_av**  and  **lon_jerk_av** with format of

`{"lat_jerk_av": dictionary_of_jerk_av, "lon_jerk_av": dictionary_of_jerk_av}`

The dictionary_of_jerk_av has format of

`{jerk: {av: stability_scroe, ...}, ...}`

## Stability Analysis on Dreamland

#### Usage

`python dreamland.py ads_bag_file`

#### Output format

The outputs contains two plots **lat_jerk_av**  and  **lon_jerk_av** with format of

`{"lat_jerk_av": dictionary_of_jerk_av, "lon_jerk_av": dictionary_of_jerk_av}`

The dictionary_of_jerk_av has format of

`{jerk: {av: cnt_of_frame, ...}, ...}`