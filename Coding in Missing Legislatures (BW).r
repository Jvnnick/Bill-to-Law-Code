# This code codes for committee members and chairs for Baden-Württemberg's Environmental Committee in the 10th, 11th, 12th, and 13th legislatures. 
# The information was missing from Fetscher et al.'s dataset, so I acquired it from Baden-Württemberg's parliamentary archival team via email 
# I then coded the data into the dataset to have more viable legislatures for the analysis.

load("~/cinc_df_julia.rda")

ls()
str(cinc_df_final)
str(cinc_df_final$committeemember_umwelt)
head(cinc_df_final$committeemember_umwelt)

library(dplyr)

# Define prefixes
prefixes <- c("committeemember_")

# Define suffixes
suffixes <- c(
  "wirtschaft", "finanzen", "recht", "landwirtschaft", "familie", "migration", "gesundheit", "arbeit",
  "sozial", "bildung", "forschung", "infrastruktur", "inneres",
  "bau", "kultur", "tourismus", "europa", "hauptausschuss", "petitionssausschuss"
)

# Generate all variable names
vars_to_remove <- unlist(lapply(prefixes, function(prefix) {
  paste0(prefix, suffixes)
}))

# Remove variables from the dataset
cinc_df_final <- cinc_df_final %>%
  select(-all_of(vars_to_remove))



# Legislature 10
cincids_leg_10_member <- c(
  "BW_Decker_Rudolf_1934", "BW_Göbel_Karl_1936", "BW_Haas_Alfred_1950",
  "BW_Hodapp_Felix_1926", "BW_Lorenz_Hans_1954", "BW_Oettinger_Günther H._1953",
  "BW_Scheuermann_Winfried_1938", "BW_Sieber_Michael_1947", "BW_Wendt_Ulrich_1945",
  "BW_Brinkmann_Ulrich_1942", "BW_Caroli_Walter_1942", "BW_Drexler_Wolfgang_1946",
  "BW_Kipfer_Birgit_1943", "BW_Maurer_Ulrich_1948", "BW_Seltenreich_Rolf_1948",
  "BW_Rochlitz_Jürgen_1937", "BW_Scharf_Bernhard_1936", "BW_Döring_Walter_1954",
  "BW_Kretschmann_Winfried_1948", "BW_Schöttle_Ventur_1929"
)

cincids_leg_10_chair <- c("BW_Decker_Rudolf_1934")
cincids_leg_10_dpc <- c("BW_Drexler_Wolfgang_1946")

# Legislature 11
cincids_leg_11_member <- c(
  "BW_Göbel_Karl_1936", "BW_Haas_Alfred_1950", "BW_Hauk_Peter_1960",
  "BW_Lorenz_Hans_1954", "BW_Müller_Ulrich_1944", "BW_Scheuermann_Winfried_1938",
  "BW_Sieber_Michael_1947", "BW_Brinkmann_Ulrich_1942", "BW_Caroli_Walter_1942",
  "BW_Drexler_Wolfgang_1946", "BW_Schmiedel_Claus_1951", "BW_Weyrosta_Claus_1925",
  "BW_Bühler_Rudolf_1939", "BW_Kuhn_Fritz_1955", "BW_Kiel_Friedrich-Wilhelm_1934"
)

cincids_leg_11_chair <- c("BW_Brinkmann_Ulrich_1942", "BW_Weyrosta_Claus_1925")
cincids_leg_11_dpc <- c("BW_Haas_Alfred_1950")

# Legislature 12
cincids_leg_12_member <- c(
  "BW_Behringer_Ernst_1942", "BW_Gräßle_Inge_1961", "BW_Hauk_Peter_1960",
  "BW_Heinz_Hans_1951", "BW_Mappus_Stefan_1966", "BW_Scheffold_Gerd_1954",
  "BW_Scheuermann_Winfried_1938", "BW_Seimetz_Hermann_1938", "BW_Steim_Hans-Jochem_1942",
  "BW_Zeiher_Martin_1952", "BW_Brechtken_Rainer_1945", "BW_Caroli_Walter_1942",
  "BW_Drexler_Wolfgang_1946", "BW_Göschel_Helmut_1944", "BW_Staiger_Wolfgang_1947",
  "BW_Stolz_Gerhard_1946", "BW_Kretschmann_Winfried_1948", "BW_Glück_Horst_1940",
  "BW_Freudenberg_Hans_1955", "BW_Eigenthaler_Egon_1938", "BW_Krisch_Wolfram_1934",
  "BW_Birk_Dietrich_1967", "BW_Wabro_Gustav_1933", "BW_Fauser_Beate_1949",
  "BW_Hehn_Karl_1940"
)

cincids_leg_12_chair <- c("BW_Kretschmann_Winfried_1948")
cincids_leg_12_dpc <- c("BW_Scheffold_Gerd_1954")

# Legislature 13
cincids_leg_13_member <- c(
  "BW_Behringer_Ernst_1942", "BW_Gurr-Hirsch_Friedlinde_1954", "BW_Hauk_Peter_1960",
  "BW_Hillebrand_Dieter_1951", "BW_Klenk_Wilfried_1959", "BW_Röhm_Karl-Wilhelm_1951",
  "BW_Schebesta_Volker_1971", "BW_Scheuermann_Winfried_1938", "BW_Steim_Hans-Jochem_1942",
  "BW_Caroli_Walter_1942", "BW_Göschel_Helmut_1944", "BW_Haller_Hans-Martin_1949",
  "BW_Kaufmann_Gunter_1944", "BW_Knapp_Thomas_1959", "BW_Schmidt-Kühner_Regina_1955",
  "BW_Staiger_Wolfgang_1947", "BW_Berroth_Heiderose_1947", "BW_Palmer_Boris_1972",
  "BW_Dederer_Heike_1969", "BW_Hitzler_Bernd_1957", "BW_Müller_Ulrich_1944"
)

cincids_leg_13_chair <- c("BW_Caroli_Walter_1942")
cincids_leg_13_dpc <- c("BW_Steim_Hans-Jochem_1942")

# Updated function to compare 'legislature' as a string
update_role_string_leg <- function(data, role_var, cincid_list, legislature_str) {
  data <- data %>%
    mutate(
      !!sym(role_var) := ifelse(
        (!!sym(role_var) == 0 | is.na(!!sym(role_var))) & 
          !!sym("legislature") == legislature_str & 
          !!sym("cincid") %in% cincid_list,
        1,
        !!sym(role_var)
      )
    )
  return(data)
}

# Apply the update for each legislature with roles as strings
# Example for Legislature 10
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeemember_umwelt", cincids_leg_10_member, "10")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeechair_umwelt", cincids_leg_10_chair, "10")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeedpchair_umwelt", cincids_leg_10_dpc, "10")

# Repeat for each legislature:
# Legislature 11
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeemember_umwelt", cincids_leg_11_member, "11")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeechair_umwelt", cincids_leg_11_chair, "11")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeedpchair_umwelt", cincids_leg_11_dpc, "11")

# Legislature 12
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeemember_umwelt", cincids_leg_12_member, "12")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeechair_umwelt", cincids_leg_12_chair, "12")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeedpchair_umwelt", cincids_leg_12_dpc, "12")

# Legislature 13
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeemember_umwelt", cincids_leg_13_member, "13")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeechair_umwelt", cincids_leg_13_chair, "13")
cinc_df_final <- update_role_string_leg(cinc_df_final, "committeedpchair_umwelt", cincids_leg_13_dpc, "13")


table(cinc_df_final$committeemember_umwelt)
filter(cinc_df_final, legislature == 10 & cincid == "BW_Decker_Rudolf_1934")

test_cincids <- c("BW_Decker_Rudolf_1934", "BW_Göbel_Karl_1936")

cinc_df_final %>%
  filter(cincid %in% test_cincids) %>%
  select(cincid, legislature, committeemember_umwelt, committeechair_umwelt, committeedpchair_umwelt)


filter(cinc_df_final, legislature == 10) %>%
  select(cincid) %>%
  distinct()

cinc_df_final %>%
  filter(cincid == "BW_Decker_Rudolf_1934" & legislature == "10") %>%
  print(n = Inf)

row_data <- cinc_df_final %>%
  filter(cincid == "BW_Decker_Rudolf_1934" & legislature == "10")
print(row_data, n = Inf)

# Check the structure of your dataset to see variable types
str(cinc_df_final)

# Specifically check the classes of the variables:
cat("committeemember_umwelt class:", class(cinc_df_final$committeemember_umwelt), "\n")
cat("committeeedpchair_umwelt class:", class(cinc_df_final$committeedpchair_umwelt), "\n")
cat("committeechair_umwelt class:", class(cinc_df_final$committeechair_umwelt), "\n")

filter(cinc_df_final, cincid == "BW_Decker_Rudolf_1934" & legislature == "10")

# List of cincid values to verify
test_cincids <- c(
  "BW_Decker_Rudolf_1934",
  "BW_Göbel_Karl_1936",
  "BW_Haas_Alfred_1950",
  "BW_Lorenz_Hans_1954",
  "BW_Scheuermann_Winfried_1938",
  "BW_Weyrosta_Claus_1925",
  "BW_Brinkmann_Ulrich_1942",
  "BW_Kretschmann_Winfried_1948"
  # Add more as desired
)

# Filter relevant rows
verification <- cinc_df_final %>%
  filter(cincid %in% test_cincids) %>%
  select(cincid, legislature, committeemember_umwelt, committeedpchair_umwelt, committeechair_umwelt)

# Print the filtered data
print(verification)

# Loop through each row for detailed inspection
for (row in 1:nrow(verification)) {
  cat("CINID:", verification$cincid[row], "\n")
  cat("Legislature:", verification$legislature[row], "\n")
  cat("committeemember_umwelt:", verification$committeemember_umwelt[row], "\n")
  cat("committeechair_umwelt:", verification$committeechair_umwelt[row], "\n")
  cat("committeedpchair_umwelt:", verification$committeedpchair_umwelt[row], "\n\n")
}


gender_ratio <- cinc_df_final %>%
  filter(committeemember_umwelt %in% c(1, 2)) %>%
  group_by(state, legislature) %>%
  summarize(
    total_members = n(),
    female_members = sum(sex == 1, na.rm = TRUE)
  ) %>%
  mutate(
    female_ratio = female_members / total_members
  ) %>%
  select(state, legislature, female_ratio)

print(gender_ratio, n = nrow(gender_ratio))

save(cinc_df_final, file = "cinc_df_julia.rda")