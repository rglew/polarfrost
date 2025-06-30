# Lets get all our imports done here
from datetime import datetime
import polars as pl
import pandas as pd
import time
import psutil
from pyspark.sql.functions import date_format, coalesce, col, lit, to_date

# Add step timing
start_time = time.time()

# Read the existing table into a Spark DataFrame
source_table = "silver.altiusinsights.client_demographics_appointments"
#df = spark.sql(f"SELECT * FROM {source_table}")

df = spark.sql("SELECT * FROM silver.altiusinsights.client_demographics_appointments c INNER JOIN silver.altiusinsights.flatorganisations o on c.client_flat_org_id = o.flat_org_id WHERE o.org_status = 'ACTIVE';")

# Add the appointment_month column in MMYYYY format, handling null dates
df = df.withColumn(
    "appointment_month",
    date_format(
        coalesce(
            col("appointment_date"),  # Use the date if not null
            to_date(lit("1970-01-01"))  # Default date if null
        ),
        "MMyyyy"
    )
).withColumn("widget_key", col("client_flat_org_id"))  # Add widget_key as copy of client_flat_org_id

print(f"Reading took {time.time() - start_time:.2f} seconds")

# Add step timing
df_conv_time = time.time()

# Convert the Spark DataFrame to a Pandas DataFrame
pdf = df.toPandas()

# Free up memory from the Spark DataFrame
df.unpersist()
del df

# Convert the Pandas DataFrame to a Polars LazyFrame
pldf = pl.from_pandas(pdf).lazy()

# Free up memory from the Pandas DataFrame
del pdf

# Configuration - top level
GROUP_COLS = [
    "client_flat_org_id",
    "appointment_month",
    "demographic_category_id",
    "demographic_response_id"
]

K_ANONYMITY = 10
DEBUG = True

result, elapsed, mem_used, row_count = anonymize_and_measure_performance(
    lazy_df=pldf,
    group_cols=GROUP_COLS,
    k=K_ANONYMITY,
    debug=DEBUG,
    schema=pldf.schema  # Pass the schema for validation
)
print(f"Converting took {time.time() - df_conv_time:.2f} seconds")
