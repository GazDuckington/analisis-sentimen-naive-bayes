import twint

# Configure
c = twint.Config()

c.Limit = 1000
c.Store_csv = True
c.Lang = "id"
c.Output = 'tweets.csv'

c.Search = "trump"

# Run
twint.run.Search(c)
