import snowflake.connector

SNOWFLAKE_USER_EMAIL="<your_email_addres>"
SNOWFLAKE_WAREHOUSE="<your_warehouse>"
SNOWFLAKE_DATABASE="<your_database>"
SNOWFLAKE_ACCOUNT='<your_account>'
SNOWFLAKE_ROLE='<your_role>'

class SnowflakeHelper:
    """
    Simple helper class for Snowflake for demo purposes.

    Attributes:
        ctx (snowflake.connector.connection): A connection to Snowflake.
    
        Functions:
            get_sf_cursor: This function returns a connection and cursor to Snowflake and a cursor using browser authentication
            execute_sf_query: This function executes a query on Snowflake and returns the results in a pandas dataframe.
            execute_sf_query_to_df: This function executes a query on Snowflake and returns the results in a pandas dataframe

    """ 
    def __init__(self):
        self.ctx = snowflake.connector.connect(user=SNOWFLAKE_USER_EMAIL, 
                                      account=SNOWFLAKE_ACCOUNT,
                                      authenticator='externalbrowser')

    def get_sf_cursor(self):
        """
        This function returns a connection and cursor to Snowflake and a cursor using browser authentication

        Returns:
            snowflake.connector.connection: A connection to Snowflake.
        """
        cs = self.ctx.cursor()
        cs.execute(f"USE ROLE {SNOWFLAKE_ROLE}")
        cs.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        cs.execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE}") 

        return cs

    
    def execute_sf_query(self, query):
        """
        This function executes a query on Snowflake and returns the results in a cursor

        Args:
            query (str): The query to execute on Snowflake.

        Returns:
            snowflake.connector.cursor: The results of the query in a cursor.
        """
        # get a connection and cursor
        cs = self.get_sf_cursor()
        
        # execute the query
        allrows = cs.execute(query)

        return allrows


    def execute_sf_query_to_df(self, query):
        """
        This function executes a query on Snowflake and returns the results in a pandas dataframe.

        Args:
            query (str): The query to execute on Snowflake.

        Returns:
            pandas.DataFrame: The results of the query in a pandas dataframe.
        """
        # get a connection and cursor
        cs = self.get_sf_cursor()

        # execute the query
        cs.execute(query)

        df = cs.fetch_pandas_all()

        return df
