# Databricks notebook source
# MAGIC %md
# MAGIC # Data-Intensive Programming - Group assignment
# MAGIC
# MAGIC This is the **Python** version of the assignment. Switch to the Scala version, if you want to do the assignment in Scala.
# MAGIC
# MAGIC In all tasks, add your solutions to the cells following the task instructions. You are free to add new cells if you want.<br>
# MAGIC The example outputs, and some additional hints are given in a separate notebook in the same folder as this one.
# MAGIC
# MAGIC Don't forget to **submit your solutions to Moodle** once your group is finished with the assignment.
# MAGIC
# MAGIC ## Basic tasks (compulsory)
# MAGIC
# MAGIC There are in total nine basic tasks that every group must implement in order to have an accepted assignment.
# MAGIC
# MAGIC The basic task 1 is a separate task, and it deals with video game sales data. The task asks you to do some basic aggregation operations with Spark data frames.
# MAGIC
# MAGIC The other basic coding tasks (basic tasks 2-8) are all related and deal with data from [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) that contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches in five European leagues during the season 2017-18. The tasks ask you to calculate the results of the matches based on the given data as well as do some further calculations. Special knowledge about football or the leagues is not required, and the task instructions should be sufficient in order to gain enough context for the tasks.
# MAGIC
# MAGIC Finally, the basic task 9 asks some information on your assignment working process.
# MAGIC
# MAGIC ## Advanced tasks (optional)
# MAGIC
# MAGIC There are in total of four advanced tasks that can be done to gain some course points. Despite the name, the advanced tasks may or may not be harder than the basic tasks.
# MAGIC
# MAGIC The advanced task 1 asks you to do all the basic tasks in an optimized way. It is possible that you gain some points from this without directly trying by just implementing the basic tasks efficiently. Logic errors and other issues that cause the basic tasks to give wrong results will be taken into account in the grading of the first advanced task. A maximum of 2 points will be given based on advanced task 1.
# MAGIC
# MAGIC The other three advanced tasks are separate tasks and their implementation does not affect the grade given for the advanced task 1.<br>
# MAGIC Only two of the three available tasks will be graded and each graded task can provide a maximum of 2 points to the total.<br>
# MAGIC If you attempt all three tasks, clearly mark which task you want to be used in the grading. Otherwise, the grader will randomly pick two of the tasks and ignore the third.
# MAGIC
# MAGIC Advanced task 2 continues with the football data and contains further questions that are done with the help of some additional data.<br>
# MAGIC Advanced task 3 deals with some image data and the questions are mostly related to the colors of the pixels in the images.<br>
# MAGIC Advanced task 4 asks you to do some classification related machine learning tasks with Spark.
# MAGIC
# MAGIC It is possible to gain partial points from the advanced tasks. I.e., if you have not completed the task fully but have implemented some part of the task, you might gain some appropriate portion of the points from the task. Logic errors, very inefficient solutions, and other issues will be taken into account in the task grading.
# MAGIC
# MAGIC ## Assignment grading
# MAGIC
# MAGIC Failing to do the basic tasks, means failing the assignment and thus also failing the course!<br>
# MAGIC "A close enough" solutions might be accepted => even if you fail to do some parts of the basic tasks, submit your work to Moodle.
# MAGIC
# MAGIC Accepted assignment submissions will be graded from 0 to 6 points.
# MAGIC
# MAGIC The maximum grade that can be achieved by doing only the basic tasks is 2/6 points (through advanced task 1).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Short summary
# MAGIC
# MAGIC ##### Minimum requirements (points: 0-2 out of maximum of 6):
# MAGIC
# MAGIC - All basic tasks implemented (at least in "a close enough" manner)
# MAGIC - Moodle submission for the group
# MAGIC
# MAGIC ##### For those aiming for higher points (0-6):
# MAGIC
# MAGIC - All basic tasks implemented
# MAGIC - Optimized solutions for the basic tasks (advanced task 1) (0-2 points)
# MAGIC - Two of the other three advanced tasks (2-4) implemented
# MAGIC     - Clearly marked which of the two tasks should be graded
# MAGIC     - Each graded advanced task will give 0-2 points
# MAGIC - Moodle submission for the group

# COMMAND ----------

# import statements for the entire notebook
# add anything that is required here

import re
from typing import Dict, List, Tuple

from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import DoubleType, StructField, StructType
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import format_string
from pyspark.sql.functions import col, array_contains, count, sum

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 1 - Video game sales data
# MAGIC
# MAGIC The CSV file `assignment/sales/video_game_sales.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains video game sales data (based on [https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom](https://www.kaggle.com/datasets/patkle/video-game-sales-data-from-vgchartzcom)).
# MAGIC
# MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
# MAGIC
# MAGIC Using the data, find answers to the following:
# MAGIC
# MAGIC - Which publisher has the highest total sales in video games in North America considering games released in years 2006-2015?
# MAGIC - How many titles in total for this publisher do not have sales data available for North America considering games released in years 2006-2015?
# MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2006-2015, what are the total sales, in North America and globally, for each year?
# MAGIC     - I.e., what are the total sales (in North America and globally) for games released by this publisher in year 2006? And the same for year 2007? ...
# MAGIC

# COMMAND ----------

salesData: DataFrame = spark.read.csv("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/sales/video_game_sales.csv",header="true", inferSchema="true",sep="|")

filteredSalesData: DataFrame = salesData.filter((F.col("release_date") >= "2006-01-01") &(F.col("release_date") <= "2015-12-31"))

NAPublisherSales: DataFrame = (filteredSalesData
                             .groupBy("publisher")
                             .agg(F.sum("na_sales").alias("total_sales"))
)

bestNAPublisher: str = NAPublisherSales.sort(F.col("total_sales").desc()).first()[0]

bestNAPublisherData: DataFrame = filteredSalesData.filter(F.col("publisher") == bestNAPublisher)

titlesWithMissingSalesData: int = bestNAPublisherData.filter(F.col("na_sales").isNull()).count()
bestNAPublisherData = bestNAPublisherData.withColumn("release_year",F.year("release_date"))

bestNAPublisherSales: DataFrame = (bestNAPublisherData
                                   .groupBy(F.col("release_year").alias("year"))
                                   .agg(
                                     F.round(F.sum("na_sales"),2).alias("na_total"),
                                     F.round(F.sum("total_sales"),2).alias("global_total")
                                   )
                                   .sort(F.col("year").asc())
)

print(f"The publisher with the highest total video game sales in North America is: '{bestNAPublisher}'")
print(f"The number of titles with missing sales data for North America: {titlesWithMissingSalesData}")
print("Sales data for the publisher:")
bestNAPublisherSales.show()

#salesData.show()
#publisherSales.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 2 - Event data from football matches
# MAGIC
# MAGIC A parquet file in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) at folder `assignment/football/events.parquet` based on [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1.
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
# MAGIC
# MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
# MAGIC
# MAGIC The team that scores more goals than their opponent wins the match.
# MAGIC
# MAGIC **Columns in the data**
# MAGIC
# MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.
# MAGIC
# MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string | The name of the competition |
# MAGIC | season | string | The season the match was played |
# MAGIC | matchId | integer | A unique id for the match |
# MAGIC | eventId | integer | A unique id for the event |
# MAGIC | homeTeam | string | The name of the home team |
# MAGIC | awayTeam | string | The name of the away team |
# MAGIC | event | string | The main category for the event |
# MAGIC | subEvent | string | The subcategory for the event |
# MAGIC | eventTeam | string | The name of the team that initiated the event |
# MAGIC | eventPlayerId | integer | The id for the player who initiated the event |
# MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
# MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
# MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
# MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
# MAGIC
# MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
# MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
# MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
# MAGIC
# MAGIC #### The task
# MAGIC
# MAGIC In this task you should load the data with all the rows into a data frame. This data frame object will then be used in the following basic tasks 3-8.

# COMMAND ----------



# COMMAND ----------

eventDF: DataFrame = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/events.parquet")
eventDF.count()
#eventDF.show(10)
#display(eventDF)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 3 - Calculate match results
# MAGIC
# MAGIC Create a match data frame for all the matches included in the event data frame created in basic task 2.
# MAGIC
# MAGIC The resulting data frame should contain one row for each match and include the following columns:
# MAGIC
# MAGIC | column name   | column type | description |
# MAGIC | ------------- | ----------- | ----------- |
# MAGIC | matchId       | integer     | A unique id for the match |
# MAGIC | competition   | string      | The name of the competition |
# MAGIC | season        | string      | The season the match was played |
# MAGIC | homeTeam      | string      | The name of the home team |
# MAGIC | awayTeam      | string      | The name of the away team |
# MAGIC | homeTeamGoals | integer     | The number of goals scored by the home team |
# MAGIC | awayTeamGoals | integer     | The number of goals scored by the away team |
# MAGIC
# MAGIC The number of goals scored for each team should be determined by the available event data.<br>
# MAGIC There are two events related to each goal:
# MAGIC
# MAGIC - One event for the player that scored the goal. This includes possible own goals.
# MAGIC - One event for the goalkeeper that tried to stop the goal.
# MAGIC
# MAGIC You need to choose which types of events you are counting.<br>
# MAGIC If you count both of the event types mentioned above, you will get double the amount of actual goals.

# COMMAND ----------

def countGoals(tags, event_team, team_name, event):
    return 1 if "Goal" in tags and event_team == team_name and event == "Save attempt" else 0



countGoals = F.udf(countGoals, IntegerType())

eventDF = (
    eventDF
    .withColumn("homeGoal", countGoals(F.col("tags"), F.col("eventTeam"), F.col("awayTeam"), F.col("event")))
    .withColumn("awayGoal", countGoals(F.col("tags"), F.col("eventTeam"), F.col("homeTeam"),F.col("event")))
)



matchDF: DataFrame = (eventDF
                      .groupBy("matchId")
                      .agg(
                        F.first("competition").alias("competition"),
                        F.first("season").alias("season"),
                        F.first("homeTeam").alias("homeTeam"),
                        F.first("awayTeam").alias("awayTeam"),
                        F.sum("homeGoal").alias("homeTeamGoals"),
                        F.sum("awayGoal").alias("awayTeamGoals")
                        )
                      
).orderBy(F.col("matchID").asc())

noGoalsCount = matchDF.filter(F.col("homeTeamGoals") == 0).filter(F.col("awayTeamGoals") == 0).count()
mostGoalsCount = (matchDF
                  .withColumn("totalGoals", F.col("homeTeamGoals") + F.col("awayTeamGoals"))
                  .agg(F.max("totalGoals").alias("mostGoals"))
)

totalGoals = (matchDF.agg(F.sum("homeTeamGoals") + F.sum("awayTeamGoals")).first()[0])


print(f"Total number of matches: {matchDF.count()}")
print(f"Matches without any goals: {noGoalsCount}")
print(f"Most goals in total in a single game: {mostGoalsCount.first()[0]}")
print(f"Total amount of goals: {totalGoals}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 4 - Calculate team points in a season
# MAGIC
# MAGIC Create a season data frame that uses the match data frame from the basic task 3 and contains aggregated seasonal results and statistics for all the teams in all leagues. While the used dataset only includes data from a single season for each league, the code should be written such that it would work even if the data would include matches from multiple seasons for each league.
# MAGIC
# MAGIC ###### Game result determination
# MAGIC
# MAGIC - Team wins the match if they score more goals than their opponent.
# MAGIC - The match is considered a draw if both teams score equal amount of goals.
# MAGIC - Team loses the match if they score fewer goals than their opponent.
# MAGIC
# MAGIC ###### Match point determination
# MAGIC
# MAGIC - The winning team gains 3 points from the match.
# MAGIC - Both teams gain 1 point from a drawn match.
# MAGIC - The losing team does not gain any points from the match.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team per league and season. It should include the following columns:
# MAGIC
# MAGIC | column name    | column type | description |
# MAGIC | -------------- | ----------- | ----------- |
# MAGIC | competition    | string      | The name of the competition |
# MAGIC | season         | string      | The season |
# MAGIC | team           | string      | The name of the team |
# MAGIC | games          | integer     | The number of games the team played in the given season |
# MAGIC | wins           | integer     | The number of wins the team had in the given season |
# MAGIC | draws          | integer     | The number of draws the team had in the given season |
# MAGIC | losses         | integer     | The number of losses the team had in the given season |
# MAGIC | goalsScored    | integer     | The total number of goals the team scored in the given season |
# MAGIC | goalsConceded  | integer     | The total number of goals scored against the team in the given season |
# MAGIC | points         | integer     | The total number of points gained by the team in the given season |

# COMMAND ----------

homeSeasonDF: DataFrame = (matchDF.groupBy("homeTeam")
                           .agg(
                             F.first("competition").alias("competition"),
                             F.first("season").alias("season"),
                             F.count("matchID").alias("homeGames"),
                             F.sum("homeTeamGoals").alias("homeGoals"),
                             F.sum("awayTeamGoals").alias("goalsConcededHome"),
                             F.sum(F.when(F.col("homeTeamGoals")>F.col("awayTeamGoals"),1).otherwise(0)).alias("homeWins"),
                             F.sum(F.when(F.col("homeTeamGoals")==F.col("awayTeamGoals"),1).otherwise(0)).alias("homeDraws"),
                             F.sum(F.when(F.col("homeTeamGoals")<F.col("awayTeamGoals"),1).otherwise(0)).alias("homeLosses"),
                             F.sum(F.when(F.col("homeTeamGoals")>F.col("awayTeamGoals"),3).otherwise(0)).alias("homeWPoints"),
                           ).withColumnRenamed("homeTeam", "team"))

awaySeasonDF: DataFrame = (matchDF.groupBy("awayTeam")
                           .agg(
                             F.count("matchID").alias("awayGames"),
                             F.sum("awayTeamGoals").alias("awayGoals"),
                             F.sum("homeTeamGoals").alias("goalsConcededaway"),
                             F.sum(F.when(F.col("homeTeamGoals")<F.col("awayTeamGoals"),1).otherwise(0)).alias("awayWins"),
                             F.sum(F.when(F.col("homeTeamGoals")==F.col("awayTeamGoals"),1).otherwise(0)).alias("awayDraws"),
                             F.sum(F.when(F.col("homeTeamGoals")>F.col("awayTeamGoals"),1).otherwise(0)).alias("awayLosses"),
                             F.sum(F.when(F.col("homeTeamGoals")<F.col("awayTeamGoals"),3).otherwise(0)).alias("awayWPoints"),
                           ).withColumnRenamed("awayTeam", "team"))

seasonDF: DataFrame = (homeSeasonDF.join(awaySeasonDF, "team", "outer")
                       .withColumn("competition",F.coalesce("competition","competition"))
                       .withColumn("season",F.coalesce("season","season"))
                       .withColumn("games", F.coalesce(F.col("homeGames"), F.lit(0)) + F.coalesce(F.col("awayGames"), F.lit(0)))
                       .withColumn("goalsScored", F.coalesce(F.col("homeGoals"), F.lit(0)) + F.coalesce(F.col("awayGoals"), F.lit(0)))
                       .withColumn("goalsConceded", F.coalesce(F.col("goalsConcededHome"), F.lit(0)) + F.coalesce(F.col("goalsConcededaway"), F.lit(0)))
                       .withColumn("draws", F.coalesce(F.col("homeDraws"), F.lit(0)) + F.coalesce(F.col("awayDraws"), F.lit(0)))
                       .withColumn("wins", F.coalesce(F.col("homeWins"), F.lit(0)) + F.coalesce(F.col("awayWins"), F.lit(0)))
                       .withColumn("losses", F.coalesce(F.col("homeLosses"), F.lit(0)) + F.coalesce(F.col("awayLosses"), F.lit(0)))
                       .withColumn("points",F.coalesce(F.col("homeWPoints"), F.lit(0)) + F.coalesce(F.col("awayWPoints"), F.lit(0)) + F.coalesce(F.col("homeDraws"), F.lit(0)) + F.coalesce(F.col("awayDraws"), F.lit(0)))
                       .select("competition", "season", "team", "games","wins", "draws","losses", "goalsScored", "goalsConceded","points")
                       )


#print(f"Total number of rows: {seasonDF.count()}")
#print(f'Teams with more that 70 points in a season: {seasonDF.filter(F.col("points") > 70).count()}')
#print(f'Lowest amount of points in a season: {seasonDF.agg(F.min("points")).first()[0]}')
#print(f"Amount of points: {seasonDF.agg(F.sum('points')).first()[0]}")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 5 - English Premier League table
# MAGIC
# MAGIC Using the season data frame from basic task 4 calculate the final league table for `English Premier League` in season `2017-2018`.
# MAGIC
# MAGIC The result should be given as data frame which is ordered by the team's classification for the season.
# MAGIC
# MAGIC A team is classified higher than the other team if one of the following is true:
# MAGIC
# MAGIC - The team has a higher number of total points than the other team
# MAGIC - The team has an equal number of points, but have a better goal difference than the other team
# MAGIC - The team has an equal number of points and goal difference, but have more goals scored in total than the other team
# MAGIC
# MAGIC Goal difference is the difference between the number of goals scored for and against the team.
# MAGIC
# MAGIC The resulting data frame should contain one row for each team.<br>
# MAGIC It should include the following columns (several columns renamed trying to match the [league table in Wikipedia](https://en.wikipedia.org/wiki/2017%E2%80%9318_Premier_League#League_table)):
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Pos         | integer     | The classification of the team |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC
# MAGIC The goal difference should be given as a string with an added `+` at the beginning if the difference is positive, similarly to the table in the linked Wikipedia article.

# COMMAND ----------

filteredDF: DataFrame = (seasonDF.filter(F.col("competition") == "English Premier League")).withColumn("gDiff", F.col("goalsScored") - F.col("goalsConceded")).orderBy(F.col("points").desc(), F.col("gDiff").desc(), F.col("goalsScored").desc()).withColumn("Position",row_number().over(Window.orderBy(F.lit(1))))
englandDF: DataFrame = (filteredDF
                        .select(
                          F.col("Position").alias("Pos"),
                          F.col("team").alias("Team"),
                          F.col("games").alias("Pld"),
                          F.col("wins").alias("W"),
                          F.col("draws").alias("D"),
                          F.col("losses").alias("L"),
                          F.col("goalsScored").alias("GF"),
                          F.col("goalsConceded").alias("GA"),
                          format_string('%+d', F.col("gDiff")).alias("GD"),
                          F.col("points").alias("Pts")
                        ))

print("English Premier League table for season 2017-2018")
englandDF.show(20, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 6: Calculate the number of passes
# MAGIC
# MAGIC This task involves going back to the event data frame and counting the number of passes each team made in each match. A pass is considered successful if it is marked as `Accurate`.
# MAGIC
# MAGIC Using the event data frame from basic task 2, calculate the total number of passes as well as the total number of successful passes for each team in each match.<br>
# MAGIC The resulting data frame should contain one row for each team in each match, i.e., two rows for each match. It should include the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | matchId     | integer     | A unique id for the match |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | season      | string      | The season |
# MAGIC | team        | string      | The name of the team |
# MAGIC | totalPasses | integer     | The total number of passes the team attempted in the match |
# MAGIC | successfulPasses | integer | The total number of successful passes made by the team in the match |
# MAGIC
# MAGIC You can assume that each team had at least one pass attempt in each match they played.

# COMMAND ----------

passDF = eventDF.filter(F.col("event") == "Pass")


matchPassDF = passDF.withColumn("AccuratePass", array_contains(F.col("tags"), "Accurate")) \
                        .groupBy("matchId", "competition", "season", "eventTeam") \
                      .agg(count("*").alias("totalPasses"), sum(col("AccuratePass").cast("int")).alias("successfulPasses")) \
                      .withColumnRenamed("eventTeam", "team").orderBy(F.col("matchId"),ascending=True)

total_rows = matchPassDF.count()

teams_high_total_passes = matchPassDF.filter(F.col("totalPasses") > 700).count()

teams_high_successful_passes = matchPassDF.filter(F.col("successfulPasses") > 600).count()


print(f"Total number of rows: {total_rows}")
print(f"Team-match pairs with more than 700 total passes: {teams_high_total_passes}")
print(f"Team-match pairs with more than 600 successful passes: {teams_high_successful_passes}")

display(matchPassDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 7: Teams with the worst passes
# MAGIC
# MAGIC Using the match pass data frame from basic task 6 find the teams with the lowest average ratio for successful passes over the season `2017-2018` for each league.
# MAGIC
# MAGIC The ratio for successful passes over a single match is the number of successful passes divided by the number of total passes.<br>
# MAGIC The average ratio over the season is the average of the single match ratios.
# MAGIC
# MAGIC Give the result as a data frame that has one row for each league-team pair with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | competition | string      | The name of the competition |
# MAGIC | team        | string      | The name of the team |
# MAGIC | passSuccessRatio | double | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the lowest ratio for passes is given first.

# COMMAND ----------

successPassRatioDF = matchPassDF.withColumn("PassSuccessRatio", (col("successfulPasses") / col("totalPasses")) * 100)

seasonPassDF = successPassRatioDF.filter(col("season") == "2017-2018") \
                                    .groupBy("competition", "team") \
                                    .agg(F.round(F.avg("PassSuccessRatio"),2).alias("PassSuccessRatio"))

windowSpec = Window.partitionBy("competition").orderBy(F.asc("PassSuccessRatio"))
rankedDF = seasonPassDF.withColumn("PassSuccessRank", F.row_number().over(windowSpec))


lowestPassSuccessRatioDF = rankedDF.filter(F.col("PassSuccessRank") == 1).drop("PassSuccessRank")

lowestPassSuccessRatioDF = lowestPassSuccessRatioDF.orderBy(F.asc("PassSuccessRatio"))

print("The teams with the lowest ratios for successful passes for each league in season 2017-2018:")
lowestPassSuccessRatioDF.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic task 8: The best teams
# MAGIC
# MAGIC For this task the best teams are determined by having the highest point average per match.
# MAGIC
# MAGIC Using the data frames created in the previous tasks find the two best teams from each league in season `2017-2018` with their full statistics.
# MAGIC
# MAGIC Give the result as a data frame with the following columns:
# MAGIC
# MAGIC | column name | column type | description |
# MAGIC | ----------- | ----------- | ----------- |
# MAGIC | Team        | string      | The name of the team |
# MAGIC | League      | string      | The name of the league |
# MAGIC | Pos         | integer     | The classification of the team within their league |
# MAGIC | Pld         | integer     | The number of games played |
# MAGIC | W           | integer     | The number of wins |
# MAGIC | D           | integer     | The number of draws |
# MAGIC | L           | integer     | The number of losses |
# MAGIC | GF          | integer     | The total number of goals scored by the team |
# MAGIC | GA          | integer     | The total number of goals scored against the team |
# MAGIC | GD          | string      | The goal difference |
# MAGIC | Pts         | integer     | The total number of points gained by the team |
# MAGIC | Avg         | double      | The average points per match gained by the team |
# MAGIC | PassRatio   | double      | The average ratio for successful passes over the season given as percentages rounded to two decimals |
# MAGIC
# MAGIC Order the data frame so that the team with the highest point average per match is given first.

# COMMAND ----------

# Optimize by pre-calculating match results and joining only once
matchResultDF = matchDF.withColumn("W", F.when(col("homeTeamGoals") > col("awayTeamGoals"), 1).otherwise(0)) \
    .withColumn("D", F.when(col("homeTeamGoals") == col("awayTeamGoals"), 1).otherwise(0)) \
    .withColumn("L", F.when(col("homeTeamGoals") < col("awayTeamGoals"), 1).otherwise(0)) \
    .withColumn("team", col("homeTeam")) \
    .withColumn("GF", col("homeTeamGoals")) \
    .withColumn("GA", col("awayTeamGoals")) \
    .union(
    matchDF.withColumn("W", F.when(col("homeTeamGoals") < col("awayTeamGoals"), 1).otherwise(0)) \
        .withColumn("D", F.when(col("homeTeamGoals") == col("awayTeamGoals"), 1).otherwise(0)) \
        .withColumn("L", F.when(col("homeTeamGoals") > col("awayTeamGoals"), 1).otherwise(0)) \
        .withColumn("team", col("awayTeam")) \
        .withColumn("GF", col("awayTeamGoals")) \
        .withColumn("GA", col("homeTeamGoals"))
)

# Aggregate match statistics
seasonStatDF = matchResultDF.groupBy("competition", "team") \
    .agg(
    count("*").alias("Pld"),  # Total matches played
    sum("W").alias("W"),  # Total wins
    sum("D").alias("D"),  # Total draws
    sum("L").alias("L"),  # Total losses
    sum("GF").alias("GF"),  # Total goals for
    sum("GA").alias("GA")  # Total goals against
)
    
# Calculate points, goal difference (GD), and point average (Avg)
seasonDF = seasonStatDF.withColumn("Pts", col("W") * 3 + col("D")) \
    .withColumn("GD", col("GF") - col("GA")) \
    .withColumn("Avg", F.round(col("Pts") / col("Pld"), 2))

# Ensure the "season" column exists in seasonPassDF (from Basic Task 7)
if "season" not in seasonPassDF.columns:
    seasonPassDF = seasonPassDF.withColumn("season", F.lit("2017-2018"))

# Join pass success ratios with team statistics (using a left join to handle potential missing data)
seasonStatsDF = seasonDF.join(
    seasonPassDF.filter(col("season") == "2017-2018").drop("season"),
    on=["competition", "team"],
    how="left"
).withColumnRenamed("competition", "League") \
    .withColumnRenamed("team", "Team") \
    .withColumnRenamed("PassSuccessRatio", "PassRatio")

# Rank teams by point average (Avg) per league
windowSpec = Window.partitionBy("League").orderBy(F.desc("Avg"), F.desc("Pts"))
rankedDF = seasonStatsDF.withColumn("Pos", row_number().over(windowSpec))

# Filter for the top 2 teams per league
bestTeamsDF = rankedDF.filter(col("Pos") <= 2)

# Select and order columns, format GD as string, round PassRatio
bestTeamsDF = bestTeamsDF.select(
    "Team", "League", "Pos", "Pld", "W", "D", "L", "GF", "GA",
    format_string("%+d", col("GD")).alias("GD"), "Pts", "Avg",
    F.round("PassRatio", 2).alias("PassRatio")
).orderBy(F.desc("Avg"), F.desc("Pts"))


# Step 8: Display the results
print("The top 2 teams for each league in season 2017-2018:")
bestTeamsDF.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Task 9: General information
# MAGIC
# MAGIC Answer **briefly** to the following questions.
# MAGIC
# MAGIC Remember that using AI and collaborating with other students outside your group is allowed as long as the usage and collaboration is documented.<br>
# MAGIC However, every member of the group should have some contribution to the assignment work.
# MAGIC
# MAGIC - Who were your group members and their contributions to the work?
# MAGIC     - Marius Ishimwe: Basic Task 1-5 and Advanced Task 2
# MAGIC     - Quang Nguyen: Basic Task 6-8 and Advanced Task 4
# MAGIC
# MAGIC - Did you use AI tools while doing the assignment?
# MAGIC     - Copilot and ChatGPT were used to double check for the right syntax to use in certain cases
# MAGIC
# MAGIC - Did you work with students outside your assignment group?
# MAGIC     - We did not work with students outside our assignment group

# COMMAND ----------

# MAGIC %md
# MAGIC ???

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced tasks
# MAGIC
# MAGIC The implementation of the basic tasks is compulsory for every group.
# MAGIC
# MAGIC Doing the following advanced tasks you can gain course points which can help in getting a better grade from the course.<br>
# MAGIC Partial solutions can give partial points.
# MAGIC
# MAGIC The advanced task 1 will be considered in the grading for every group based on their solutions for the basic tasks.
# MAGIC
# MAGIC The advanced tasks 2, 3, and 4 are separate tasks. The solutions used in these other advanced tasks do not affect the grading of advanced task 1. Instead, a good use of optimized methods can positively influence the grading of each specific task, while very non-optimized solutions can have a negative effect on the task grade.
# MAGIC
# MAGIC While you can attempt all three tasks (advanced tasks 2-4), only two of them will be graded and contribute towards the course grade.<br>
# MAGIC Mark in the following cell which tasks you want to be graded and which should be ignored.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### If you did the advanced tasks 2-4, mark here which of the two should be considered in grading:
# MAGIC
# MAGIC - Advanced task 2 should be graded: V
# MAGIC - Advanced task 3 should be graded: 
# MAGIC - Advanced task 4 should be graded: V

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 1 - Optimized and correct solutions to the basic tasks (2 points)
# MAGIC
# MAGIC Use the tools Spark offers effectively and avoid unnecessary operations in the code for the basic tasks.
# MAGIC
# MAGIC A couple of things to consider (**not** even close to a complete list):
# MAGIC
# MAGIC - Consider using explicit schemas when dealing with CSV data sources.
# MAGIC - Consider only including those columns from a data source that are actually needed.
# MAGIC - Filter unnecessary rows whenever possible to get smaller datasets.
# MAGIC - Avoid collect or similar expensive operations for large datasets.
# MAGIC - Consider using explicit caching if some data frame is used repeatedly.
# MAGIC - Avoid unnecessary shuffling (for example sorting) operations.
# MAGIC - Avoid unnecessary actions (count, etc.) that are not needed for the task.
# MAGIC
# MAGIC In addition to the effectiveness of your solutions, the correctness of the solution logic will be taken into account when determining the grade for this advanced task 1.
# MAGIC "A close enough" solution with some logic fails might be enough to have an accepted group assignment, but those failings might lower the score for this task.
# MAGIC
# MAGIC It is okay to have your own test code that would fall into category of "ineffective usage" or "unnecessary operations" while doing the assignment tasks. However, for the final Moodle submission you should comment out or delete such code (and test that you have not broken anything when doing the final modifications).
# MAGIC
# MAGIC Note, that you should not do the basic tasks again for this additional task, but instead modify your basic task code with more efficient versions.
# MAGIC
# MAGIC You can create a text cell below this one and describe what optimizations you have done. This might help the grader to better recognize how skilled your work with the basic tasks has been.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 2 - Further tasks with football data (2 points)
# MAGIC
# MAGIC This advanced task continues with football event data from the basic tasks. In addition, there are two further related datasets that are used in this task.
# MAGIC
# MAGIC A Parquet file at folder `assignment/football/matches.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about which players were involved on each match including information on the substitutions made during the match.
# MAGIC
# MAGIC Another Parquet file at folder `assignment/football/players.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains information about the player names, default roles when playing, and their birth areas.
# MAGIC
# MAGIC #### Columns in the additional data
# MAGIC
# MAGIC The match dataset (`assignment/football/matches.parquet`) has one row for each match and each row has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | matchId      | integer     | A unique id for the match |
# MAGIC | competition  | string      | The name of the league |
# MAGIC | season       | string      | The season the match was played |
# MAGIC | roundId      | integer     | A unique id for the round in the competition |
# MAGIC | gameWeek     | integer     | The gameWeek of the match |
# MAGIC | date         | date        | The date the match was played |
# MAGIC | status       | string      | The status of the match, `Played` if the match has been played |
# MAGIC | homeTeamData | struct      | The home team data, see the table below for the attributes in the struct |
# MAGIC | awayTeamData | struct      | The away team data, see the table below for the attributes in the struct |
# MAGIC | referees     | struct      | The referees for the match |
# MAGIC
# MAGIC Both team data columns have the following inner structure:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | team         | string      | The name of the team |
# MAGIC | coachId      | integer     | A unique id for the coach of the team |
# MAGIC | lineup       | array of integers | A list of the player ids who start the match on the field for the team |
# MAGIC | bench        | array of integers | A list of the player ids who start the match on the bench, i.e., the reserve players for the team |
# MAGIC | substitution1 | struct     | The first substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution2 | struct     | The second substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC | substitution3 | struct     | The third substitution the team made in the match, see the table below for the attributes in the struct |
# MAGIC
# MAGIC Each substitution structs have the following inner structure:
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerIn     | integer     | The id for the player who was substituted from the bench into the field, i.e., this player started playing after this substitution |
# MAGIC | playerOut    | integer     | The id for the player who was substituted from the field to the bench, i.e., this player stopped playing after this substitution |
# MAGIC | minute       | integer     | The minute from the start of the match the substitution was made.<br>Values of 45 or less indicate that the substitution was made in the first half of the match,<br>and values larger than 45 indicate that the substitution was made on the second half of the match. |
# MAGIC
# MAGIC The player dataset (`assignment/football/players.parquet`) has the following columns:
# MAGIC
# MAGIC | column name  | column type | description |
# MAGIC | ------------ | ----------- | ----------- |
# MAGIC | playerId     | integer     | A unique id for the player |
# MAGIC | firstName    | string      | The first name of the player |
# MAGIC | lastName     | string      | The last name of the player |
# MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
# MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
# MAGIC | foot         | string      | The stronger foot of the player |
# MAGIC
# MAGIC #### Background information
# MAGIC
# MAGIC In a football match both teams have 11 players on the playing field or pitch at the start of the match. Each team also have some number of reserve players on the bench at the start of the match. The teams can make up to three substitution during the match where they switch one of the players on the field to a reserve player. (Currently, more substitutions are allowed, but at the time when the data is from, three substitutions were the maximum.) Any player starting the match as a reserve and who is not substituted to the field during the match does not play any minutes and are not considered involved in the match.
# MAGIC
# MAGIC For this task the length of each match should be estimated with the following procedure:
# MAGIC
# MAGIC - Only the additional time added to the second half of the match should be considered. I.e., the length of the first half is always considered to be 45 minutes.
# MAGIC - The length of the second half is to be considered as the last event of the half rounded upwards towards the nearest minute.
# MAGIC     - I.e., if the last event of the second half happens at 2845 seconds (=47.4 minutes) from the start of the half, the length of the half should be considered as 48 minutes. And thus, the full length of the entire match as 93 minutes.
# MAGIC
# MAGIC A personal plus-minus statistics for each player can be calculated using the following information:
# MAGIC
# MAGIC - If a goal was scored by the player's team when the player was on the field, `add 1`
# MAGIC - If a goal was scored by the opponent's team when the player was on the field, `subtract 1`
# MAGIC - If a goal was scored when the player was a reserve on the bench, `no change`
# MAGIC - For any event that is not a goal, or is in a match that the player was not involved in, `no change`
# MAGIC - Any substitutions is considered to be done at the start of the given minute.
# MAGIC     - I.e., if the player is substituted from the bench to the field at minute 80 (minute 35 on the second half), they were considered to be on the pitch from second 2100.0 on the 2nd half of the match.
# MAGIC - If a goal was scored in the additional time of the first half of the match, i.e., the goal event period is `1H` and event time is larger than 2700 seconds, some extra considerations should be taken into account:
# MAGIC     - If a player is substituted into the field at the beginning of the second half, `no change`
# MAGIC     - If a player is substituted off the field at the beginning of the second half, either `add 1` or `subtract 1` depending on team that scored the goal
# MAGIC     - Any player who is substituted into the field at minute 45 or later is only playing on the second half of the match.
# MAGIC     - Any player who is substituted off the field at minute 45 or later is considered to be playing the entire first half including the additional time.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to use the football event data and the additional datasets to determine the following:
# MAGIC
# MAGIC - The players with the most total minutes played in season 2017-2018 for each player role
# MAGIC     - I.e., the player in Goalkeeper role who has played the longest time across all included leagues. And the same for the other player roles (Defender, Midfielder, and Forward)
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `role`: the player role
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `minutes`: the total minutes the player played during season 2017-2018
# MAGIC - The players with higher than `+65` for the total plus-minus statistics in season 2017-2018
# MAGIC     - Give the result as a data frame that has the following columns:
# MAGIC         - `player`: the full name of the player, i.e., the first name combined with the last name
# MAGIC         - `birthArea`: the birth area of the player
# MAGIC         - `role`: the player role
# MAGIC         - `plusMinus`: the total plus-minus statistics for the player during season 2017-2018
# MAGIC
# MAGIC It is advisable to work towards the target results using several intermediate steps.

# COMMAND ----------

#load file with match and player data

matchesDF = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/matches.parquet")
playersDF = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/football/players.parquet")


#display(matchesDF)
#display(playersDF)

secondHalf = eventDF.filter(F.col("eventPeriod") == "2H")

secondHalfTime = secondHalf.groupBy("matchId").agg(F.ceil(F.max("eventTime")/60).alias("secondHalfMinutes"))

matchTime = secondHalfTime.withColumn("matchLength",F.col("secondHalfMinutes")+45)
matchTimes = matchTime.select("matchId","matchLength").orderBy("matchId")

homePlayerTime = matchesDF.select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.col("homeTeamData.team").alias("playerTeam"),
    F.explode(F.col("homeTeamData.lineup")).alias("playerId"),
    F.lit(0).alias("startMinute"),
    F.lit(None).alias("endMinute")
)

awayPlayersTime = matchesDF.select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.col("awayTeamData.team").alias("playerTeam"),
    F.explode(F.col("awayTeamData.lineup")).alias("playerId"),
    F.lit(0).alias("startMinute"),
    F.lit(None).alias("endMinute")
)

lineupDF = homePlayerTime.union(awayPlayersTime)

substitutions = [
    "homeTeamData.substitution1",
    "homeTeamData.substitution2",
    "homeTeamData.substitution3",
    "awayTeamData.substitution1",
    "awayTeamData.substitution2",
    "awayTeamData.substitution3"
]

substitutionsInDF = matchesDF.select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.explode(F.array(*[F.col(col) for col in substitutions])).alias("substitution")
).select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.col("substitution.playerIn").alias("playerId"),
    F.col("substitution.minute").alias("startMinute"),
    F.lit(None).cast("integer").alias("endMinute"),
    
)

substitutionsOutDF = matchesDF.select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.explode(F.array(*[F.col(col) for col in substitutions])).alias("substitution")
).select(
    F.col("matchId"),
    F.col("competition"),
    F.col("season"),
    F.col("substitution.playerOut").alias("playerId"),
    F.lit(None).cast("integer").alias("startMinute"),
    F.col("substitution.minute").alias("endMinute"),
    
)

substitutionsDF = substitutionsInDF.union(substitutionsOutDF)

substitutionsDF = substitutionsDF.join( 
                    matchesDF.select( 
                        F.col("matchId"), 
                        F.explode(F.array( 
                            F.struct(F.col("homeTeamData.team").alias("playerTeam"), F.col("homeTeamData.substitution1.playerIn").alias("playerId")), 
                            F.struct(F.col("homeTeamData.team").alias("playerTeam"), F.col("homeTeamData.substitution2.playerIn").alias("playerId")), 
                            F.struct(F.col("homeTeamData.team").alias("playerTeam"), F.col("homeTeamData.substitution3.playerIn").alias("playerId")), 
                            F.struct(F.col("awayTeamData.team").alias("playerTeam"), F.col("awayTeamData.substitution1.playerIn").alias("playerId")), 
                            F.struct(F.col("awayTeamData.team").alias("playerTeam"), F.col("awayTeamData.substitution2.playerIn").alias("playerId")), 
                            F.struct(F.col("awayTeamData.team").alias("playerTeam"), F.col("awayTeamData.substitution3.playerIn").alias("playerId")) )).alias("exploded") ).select("matchId", "exploded.playerTeam", "exploded.playerId"), ["matchId", "playerId"] )




lineupDF = lineupDF.select("matchId", "playerId", "competition", "season", "playerTeam", "startMinute", "endMinute")
substitutionsDF = substitutionsDF.select("matchId", "playerId", "competition", "season", "playerTeam", "startMinute", "endMinute")

allPlayersDF = lineupDF.union(substitutionsDF)

playersTime = allPlayersDF.join(matchTimes, "matchId").withColumn( 
    "startMinute", F.when(F.col("startMinute").isNull(), 0).otherwise(F.col("startMinute")) 
    ).withColumn( "endMinute", F.when(F.col("endMinute").isNull(), F.when(F.col("startMinute").isNull(), 0).otherwise(F.col("matchLength"))).otherwise(F.col("endMinute")) 
    ).withColumn( "minutes", F.when(F.col("startMinute").isNull() & F.col("endMinute").isNull(), 0).otherwise(F.col("endMinute") - F.col("startMinute")) 
    ).select( "matchId", "playerId", "competition", "season", "playerTeam", "startMinute", "endMinute", "minutes" )


eventDF = eventDF.withColumn( "goalTeam", 
                    F.when(F.col("homeGoal") == 1, F.col("homeTeam")) 
                    .when(F.col("awayGoal") == 1, F.col("awayTeam")) 
                    )

playerPlusMinusDF = playersTime.join(eventDF, "matchId").withColumn( "eventTime", (F.col("eventTime") / 60) 
                        ).withColumn( "playerPlusMinus", F.when( 
                                                                (F.col("homeGoal") == 1) & (F.col("playerTeam") == F.col("goalTeam")) & (F.col("eventTime") >= F.col("startMinute")) & (F.col("eventTime") <= F.col("endMinute")), 1 
                                                                ).when( 
                                                                       (F.col("awayGoal") == 1) & (F.col("playerTeam") != F.col("goalTeam")) & (F.col("eventTime") >= F.col("startMinute")) & (F.col("eventTime") <= F.col("endMinute")), -1 
                                                                       ).otherwise(0) 
                                     ).groupBy( "eventId", "matchId", "playerId", "playerTeam", "eventTime", "goalTeam", "startMinute", "endMinute" 
                                               ).agg( 
                                                     F.sum("playerPlusMinus").alias("playerPlusMinus") 
                        ).select( "eventId", "matchId", "playerId", "playerTeam", "eventTime", "goalTeam", "startMinute", "endMinute", "playerPlusMinus" )


playersRoleDF = playersTime.join(playersDF, playersTime.playerId == playersDF.playerId).select( 
                                                                                               "role", F.concat(F.col("firstName"), F.lit(" "), F.col("lastName")).alias("player"), "birthArea", "minutes" 
                                                                                               )

totalMinutesDF = playersRoleDF.groupBy("role", "player", "birthArea").agg( F.sum("minutes").alias("totalMinutes") )

windowSpec = Window.partitionBy("role").orderBy(F.desc("totalMinutes"))

mostMinutesDF: DataFrame = totalMinutesDF.withColumn("rank", 
                                                     F.row_number().over(windowSpec)).filter(F.col("rank") == 1).select( 
                                                                                        "role", "player", "birthArea", "totalMinutes" 
                                                                                        ).orderBy("role").orderBy(F.desc("totalMinutes"))




#display(topPlayersDF)



print("The players with the most minutes played in season 2017-2018 for each player role:")
mostMinutesDF.show(truncate=False)

# COMMAND ----------


#topPlayers: DataFrame = ???



topPlayers: DataFrame = ???


#print("The players with higher than +65 for the plus-minus statistics in season 2017-2018:")
#topPlayers.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 3 - Image data and pixel colors (2 points)
# MAGIC
# MAGIC This advanced task involves loading in PNG image data and complementing JSON metadata into Spark data structure. And then determining the colors of the pixels in the images, and finding the answers to several color related questions.
# MAGIC
# MAGIC The folder `assignment/openmoji/color` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains collection of PNG images from [OpenMoji](https://openmoji.org/) project.
# MAGIC
# MAGIC The JSON Lines formatted file `assignment/openmoji/openmoji.jsonl` contains metadata about the image collection. Only a portion of the images are included as source data for this task, so the metadata file contains also information about images not considered in this task.
# MAGIC
# MAGIC #### Data description and helper functions
# MAGIC
# MAGIC The image data considered in this task can be loaded into a Spark data frame using the `image` format: [https://spark.apache.org/docs/3.5.0/ml-datasource.html](https://spark.apache.org/docs/3.5.0/ml-datasource.html). The resulting data frame contains a single column which includes information about the filename, image size as well as the binary data representing the image itself. The Spark documentation page contains more detailed information about the structure of the column.
# MAGIC
# MAGIC Instead of using the images as source data for machine learning tasks, the binary image data is accessed directly in this task.<br>
# MAGIC You are given two helper functions to help in dealing with the binary data:
# MAGIC
# MAGIC - Function `toPixels` takes in the binary image data and the number channels used to represent each pixel.
# MAGIC     - In the case of the images used in this task, the number of channels match the number bytes used for each pixel.
# MAGIC     - As output the function returns an array of strings where each string is hexadecimal representation of a single pixel in the image.
# MAGIC - Function `toColorName` takes in a single pixel represented as hexadecimal string.
# MAGIC     - As output the function returns a string with the name of the basic color that most closely represents the pixel.
# MAGIC     - The function uses somewhat naive algorithm to determine the name of the color, and does not always give correct results.
# MAGIC     - Many of the pixels in this task have a lot of transparent pixels. Any such pixel is marked as the color `None` by the function.
# MAGIC
# MAGIC With the help of the given functions it is possible to transform the binary image data to an array of color names without using additional libraries or knowing much about image processing.
# MAGIC
# MAGIC The metadata file given in JSON Lines format can be loaded into a Spark data frame using the `json` format: [https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html](https://spark.apache.org/docs/3.5.0/sql-data-sources-json.html). The attributes used in the JSON data are not described here, but are left for you to explore. The original regular JSON formatted file can be found at [https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json](https://github.com/hfg-gmuend/openmoji/blob/master/data/openmoji.json).
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC The target of the task is to combine the image data with the JSON data, determine the image pixel colors, and the find the answers to the following questions:
# MAGIC
# MAGIC - Which four images have the most colored non-transparent pixels?
# MAGIC - Which five images have the lowest ratio of colored vs. transparent pixels?
# MAGIC - What are the three most common colors in the Finnish flag image (annotation: `flag: Finland`)?
# MAGIC     - And how many percentages of the colored pixels does each color have?
# MAGIC - How many images have their most common three colors as, `Blue`-`Yellow`-`Black`, in that order?
# MAGIC - Which five images have the most red pixels among the image group `activities`?
# MAGIC     - And how many red pixels do each of these images have?
# MAGIC
# MAGIC It might be advisable to test your work-in-progress code with a limited number of images before using the full image set.<br>
# MAGIC You are free to choose your own approach to the task: user defined functions with data frames, RDDs/Datasets, or combination of both.
# MAGIC
# MAGIC Note that the currently the Python helper functions do not exactly match the Scala versions, and thus the answers to the questions might not quite match the given example results in the example output notebook.

# COMMAND ----------

# separates binary image data to an array of hex strings that represent the pixels
# assumes 8-bit representation for each pixel (0x00 - 0xff)
# with `channels` attribute representing how many bytes is used for each pixel
def toPixels(data: bytes, channels: int) -> List[str]:
    return [
        "".join([
            f"{data[index+byte]:02X}"
            for byte in range(0, channels)
        ])
        for index in range(0, len(data), channels)
    ]

# COMMAND ----------

# naive implementation of picking the name of the pixel color based on the input hex representation of the pixel
# only works for OpenCV type CV_8U (mode=24) compatible input
def toColorName(hexString: str) -> str:
    # mapping of RGB values to basic color names
    colors: Dict[Tuple[int, int, int], str] = {
        (0, 0, 0):     "Black",  (0, 0, 128):     "Blue",   (0, 0, 255):     "Blue",
        (0, 128, 0):   "Green",  (0, 128, 128):   "Green",  (0, 128, 255):   "Blue",
        (0, 255, 0):   "Green",  (0, 255, 128):   "Green",  (0, 255, 255):   "Blue",
        (128, 0, 0):   "Red",    (128, 0, 128):   "Purple", (128, 0, 255):   "Purple",
        (128, 128, 0): "Green",  (128, 128, 128): "Gray",   (128, 128, 255): "Purple",
        (128, 255, 0): "Green",  (128, 255, 128): "Green",  (128, 255, 255): "Blue",
        (255, 0, 0):   "Red",    (255, 0, 128):   "Pink",   (255, 0, 255):   "Purple",
        (255, 128, 0): "Orange", (255, 128, 128): "Orange", (255, 128, 255): "Pink",
        (255, 255, 0): "Yellow", (255, 255, 128): "Yellow", (255, 255, 255): "White"
    }

    # helper function to round values of 0-255 to the nearest of 0, 128, or 255
    def roundColorValue(value: int) -> int:
        if value < 85:
            return 0
        if value < 170:
            return 128
        return 255

    validString: bool = re.match(r"[0-9a-fA-F]{8}", hexString) is not None
    if validString:
        # for OpenCV type CV_8U (mode=24) the expected order of bytes is BGRA
        blue: int = roundColorValue(int(hexString[0:2], 16))
        green: int = roundColorValue(int(hexString[2:4], 16))
        red: int = roundColorValue(int(hexString[4:6], 16))
        alpha: int = int(hexString[6:8], 16)

        if alpha < 128:
            return "None"  # any pixel with less than 50% opacity is considered as color "None"
        return colors[(red, green, blue)]

    return "None"  # any input that is not in valid format is considered as color "None"

# COMMAND ----------

imageDF = spark.read.format("image").load("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/openmoji/color/*.png")
metadataDF = spark.read.json("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/openmoji/metadata/openmoji.jsonl")

combined_df = (
    imageDF.select("image.origin", "image.data", "image.nChannels")
    .withColumnRenamed("origin", "filepath")
    .withColumn(
        "filename",
        F.regexp_extract(F.col("filepath"), r"([^/]+)\.png$", 1),
    )
    .join(metadataDF, F.col("filename") == metadataDF["hexcode"], "inner")
    .withColumn("pixels", F.udf(toPixels, ArrayType(StringType()))(F.col("data"), F.col("nChannels")))
    .withColumn("colors", F.explode_outer(col("pixels")).alias("pixel"))
    .withColumn("color_name", F.udf(toColorName, StringType())(F.col("pixels")))
    .drop("data", "nChannels", "pixel", "keywords", "skintone", "group", "subgroups", "tags", "category", "order"))


# The annotations for the four images with the most colored non-transparent pixels
mostColoredPixels: List[str] = (
    combined_df.filter(F.col("color_name") != "None")
    .groupBy("annotation")
    .agg(F.count("color_name").alias("colored_pixel_count"))
    .orderBy(F.desc("colored_pixel_count"))
    .limit(4)
    .select("annotation")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print("The annotations for the four images with the most colored non-transparent pixels:")
for image in mostColoredPixels:
    print(f"- {image}")
print("============================================================")


# The annotations for the five images having the lowest ratio of colored vs. transparent pixels
leastColoredPixels: List[str] = (
    combined_df.groupBy("annotation")
    .agg(
        count("color_name").alias("total_pixels"),
        count(when(col("color_name") != "None", 1)).alias("colored_pixels"),
    )
    .withColumn(
        "colored_ratio",
        when(col("total_pixels") > 0, col("colored_pixels") / col("total_pixels")).otherwise(0.0),
    )
    .orderBy("colored_ratio")
    .limit(5)
    .select("annotation")
    .rdd.flatMap(lambda x: x)
    .collect()
)

print("The annotations for the five images having the lowest ratio of colored vs. transparent pixels:")
for image in leastColoredPixels:
    print(f"- {image}")

# COMMAND ----------

# The three most common colors in the Finnish flag image:
finnishFlagColors: List[str] = ???

# The percentages of the colored pixels for each common color in the Finnish flag image:
finnishColorShares: List[float] = ???

print("The colors and their percentage shares in the image for the Finnish flag:")
for color, share in zip(finnishFlagColors, finnishColorShares):
    print(f"- color: {color}, share: {share}")
print("============================================================")


# The number of images that have their most common three colors as, Blue-Yellow-Black, in that exact order:
blueYellowBlackCount: int = ???

print(f"The number of images that have, Blue-Yellow-Black, as the most common colors: {blueYellowBlackCount}")

# COMMAND ----------

# The annotations for the five images with the most red pixels among the image group activities:
redImageNames: List[str] = ???

# The number of red pixels in the five images with the most red pixels among the image group activities:
redPixelAmounts: List[int] = ???

print("The annotations and red pixel counts for the five images with the most red pixels among the image group 'activities':")
for color, pixel_count in zip(redImageNames, redPixelAmounts):
    print(f"- {color} (red pixels: {pixel_count})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Advanced Task 4 - Machine learning tasks (2 points)
# MAGIC
# MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the [ProCem](https://www.senecc.fi/projects/procem-2) research project is used as the training and test data. Similar data in a slightly different format was used in the first tasks of weekly exercise 3.
# MAGIC
# MAGIC The folder `assignment/energy/procem_13m.parquet` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2024-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2024gen2/path/shared/etag/%220x8DBB0695B02FFFE%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None) contains the time series data in Parquet format.
# MAGIC
# MAGIC #### Data description
# MAGIC
# MAGIC The dataset contains time series data from a period of 13 months (from the beginning of May 2023 to the end of May 2024). Each row contains the average of the measured values for a single minute. The following columns are included in the data:
# MAGIC
# MAGIC | column name        | column type   | description |
# MAGIC | ------------------ | ------------- | ----------- |
# MAGIC | time               | long          | The UNIX timestamp in second precision |
# MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
# MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
# MAGIC | wind_speed         | double        | The wind speed measured by the weather station on top of Sähkötalo (`m/s`) |
# MAGIC | power_tenants      | double        | The total combined electricity power used by the tenants on Kampusareena (`W`) |
# MAGIC | power_maintenance  | double        | The total combined electricity power used by the building maintenance systems on Kampusareena (`W`) |
# MAGIC | power_solar_panels | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
# MAGIC | electricity_price  | double        | The market price for electricity in Finland (`€/MWh`) |
# MAGIC
# MAGIC There are some missing values that need to be removed before using the data for training or testing. However, only the minimal amount of rows should be removed for each test case.
# MAGIC
# MAGIC ### Tasks
# MAGIC
# MAGIC - The main task is to train and test a machine learning model with [Random forest classifier](https://spark.apache.org/docs/3.5.0/ml-classification-regression.html#random-forests) in six different cases:
# MAGIC     - Predict the month (1-12) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the month (1-12) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the month (1-12) using all seven measurements (weather values, power values, and price) as input
# MAGIC     - Predict the hour of the day (0-23) using the three weather measurements (temperature, humidity, and wind speed) as input
# MAGIC     - Predict the hour of the day (0-23) using the three power measurements (tenants, maintenance, and solar panels) as input
# MAGIC     - Predict the hour of the day (0-23) using all seven measurements (weather values, power values, and price) as input
# MAGIC - For each of the six case you are asked to:
# MAGIC     1. Clean the source dataset from rows with missing values.
# MAGIC     2. Split the dataset into training and test parts.
# MAGIC     3. Train the ML model using a Random forest classifier with case-specific input and prediction.
# MAGIC     4. Evaluate the accuracy of the model with Spark built-in multiclass classification evaluator.
# MAGIC     5. Further evaluate the accuracy of the model with a custom build evaluator which should do the following:
# MAGIC         - calculate the percentage of correct predictions
# MAGIC             - this should correspond to the accuracy value from the built-in accuracy evaluator
# MAGIC         - calculate the percentage of predictions that were at most one away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be `4`, `5`, or `6`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be `12`, `1`, or `2`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be `11`, `12`, or `1`
# MAGIC         - calculate the percentage of predictions that were at most two away from the correct predictions taking into account the cyclic nature of the month and hour values:
# MAGIC             - if the correct month value was `5`, then acceptable predictions would be from `3` to `7`
# MAGIC             - if the correct month value was `1`, then acceptable predictions would be from `11` to `12` and from `1` to `3`
# MAGIC             - if the correct month value was `12`, then acceptable predictions would be from `10` to `12` and from `1` to `2`
# MAGIC         - calculate the average probability the model predicts for the correct value
# MAGIC             - the probabilities for a single prediction can be found from the `probability` column after the predictions have been made with the model
# MAGIC - As the final part of this advanced task, you are asked to do the same experiments (training+evaluation) with two further cases of your own choosing:
# MAGIC     - you can decide on the input columns yourself
# MAGIC     - you can decide the predicted attribute yourself
# MAGIC     - you can try some other classifier other than the random forest one if you want
# MAGIC
# MAGIC In all cases you are free to choose the training parameters as you wish.<br>
# MAGIC Note that it is advisable that while you are building your task code to only use a portion of the full 13-month dataset in the initial experiments.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import DataFrame
from typing import List

# Load the dataset
df = spark.read.parquet("abfss://shared@tunics320f2024gen2.dfs.core.windows.net/assignment/energy/procem_13m.parquet")

# Helper function to clean and prepare data
def prepare_data(df: DataFrame, input_cols: List[str], label_col: str) -> DataFrame:
    cleaned_df = df.na.drop(subset=input_cols + [label_col])  # Drop rows with missing values
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    assembled_df = assembler.transform(cleaned_df)
    return assembled_df

# Helper function to train and evaluate a model
def train_and_evaluate(classifier, train_df, test_df, label_col, classifier_name, input_cols_str):

    model = classifier.fit(train_df)
    predictions = model.transform(test_df)

    # Built-in evaluator
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Training a '{classifier_name}' model to predict '{label_col}' based on input '{input_cols_str}'.")
    print(f"The accuracy of the model is {accuracy}")

    return predictions, accuracy

# Helper function for custom evaluation metrics
def custom_evaluate(predictions_df, label_col, max_val):
    def within_range(prediction, label, tolerance):
        diff = F.abs(prediction - label)
        return (diff <= tolerance) | (max_val - diff <= tolerance)

    predictions_df = predictions_df.withColumn(label_col, F.col(label_col).cast(IntegerType()))
    predictions_df = predictions_df.withColumn("prediction", F.col("prediction").cast(IntegerType()))

    correct_count = predictions_df.filter(F.col("prediction") == F.col(label_col)).count()
    within_one_count = predictions_df.filter(within_range(F.col("prediction"), F.col(label_col), 1)).count()
    within_two_count = predictions_df.filter(within_range(F.col("prediction"), F.col(label_col), 2)).count()
    
    total_count = predictions_df.count()

    correct = (correct_count / total_count) * 100
    within_one = (within_one_count / total_count) * 100
    within_two = (within_two_count / total_count) * 100

    # Calculate average probability for correct predictions
    probability_rdd = predictions_df.select("probability", label_col, "prediction").rdd
    prob_correct_sum = probability_rdd.map(lambda row: float(row["probability"][int(row[label_col]) -1]) if int(row['prediction']) == int(row[label_col]) else 0.0).sum()

    avg_prob = (prob_correct_sum / correct_count) if correct_count > 0 else 0.0

    return correct, within_one, within_two, avg_prob

# Define cases for model training
cases = [
    {
        "label": "month",
        "input": ["temperature", "humidity", "wind_speed"],
        "max_val": 12  # Months are 1-12
    },
    {
        "label": "month",
        "input": ["power_tenants", "power_maintenance", "power_solar_panels"],
        "max_val": 12
    },
    {
        "label": "month",
        "input": ["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"],
        "max_val": 12
    },
    {
        "label": "hour",
        "input": ["temperature", "humidity", "wind_speed"],
        "max_val": 24  # Hours are 0-23
    },
    {
        "label": "hour",
        "input": ["power_tenants", "power_maintenance", "power_solar_panels"],
        "max_val": 24
    },
    {
        "label": "hour",
        "input": ["temperature", "humidity", "wind_speed", "power_tenants", "power_maintenance", "power_solar_panels", "electricity_price"],
        "max_val": 24
    },
]

# Train and evaluate models for each case
results = []
for case in cases:
    df = df.withColumn("timestamp", F.from_unixtime("time"))  # Convert 'time' to 'timestamp'
    df = df.withColumn("month", F.month("timestamp")).withColumn("hour", F.hour("timestamp"))
    prepared_df = prepare_data(df, case["input"], case["label"])
    train_df, test_df = prepared_df.randomSplit([0.8, 0.2], seed=1)
    input_cols_str = ",".join(case['input'])

    classifier = RandomForestClassifier(labelCol=case["label"], featuresCol="features", seed=1)
    predictions, accuracy = train_and_evaluate(classifier, train_df, test_df, case["label"], "RandomForest", input_cols_str)

    correct, within_one, within_two, avg_prob = custom_evaluate(predictions, case["label"], case['max_val'])
    
    results.append({
        "classifier": "RandomForest",
        "input": input_cols_str,
        "label": case["label"],
        "correct": correct,
        "within_one": within_one,
        "within_two": within_two,
        "avg_prob": avg_prob
    })

# Create a DataFrame from the results
results_df = spark.createDataFrame(results)
results_df = results_df.withColumn("correct", F.round(F.col("correct"), 2))
results_df = results_df.withColumn("within_one", F.round(F.col("within_one"), 2))
results_df = results_df.withColumn("within_two", F.round(F.col("within_two"), 2))
results_df = results_df.withColumn("avg_prob", F.round(F.col("avg_prob"), 4))
print("Gathering the asked additional accuracy test results for the previous models into a data frame:")
display(results_df)

# COMMAND ----------

# Example of additional model training (you should do different cases)
df = df.withColumn("dayofweek", F.dayofweek("timestamp"))
additional_cases = [
    {
        "label": "dayofweek",
        "input": ["power_tenants", "power_maintenance", "electricity_price"],
        "classifier": RandomForestClassifier(labelCol="dayofweek", featuresCol="scaledFeatures", seed=1),
        "max_val": 7
    },
    {
        "label": "dayofweek",
        "input": ["power_tenants", "power_maintenance", "electricity_price"],
        "classifier": NaiveBayes(featuresCol="scaledFeatures", labelCol="dayofweek"),
        "max_val": 7
    }
]

additional_results = []

for case in additional_cases:
    prepared_df = prepare_data(df, case["input"], case["label"])
    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    scalerModel = scaler.fit(prepared_df)
    scaledData = scalerModel.transform(prepared_df)
    train_df, test_df = scaledData.randomSplit([0.8, 0.2], seed=1)

    input_cols_str = ",".join(case["input"])

    predictions, accuracy = train_and_evaluate(case["classifier"], train_df, test_df, case["label"], type(case["classifier"]).__name__, input_cols_str)

    correct, within_one, within_two, avg_prob = custom_evaluate(predictions, case["label"], case["max_val"])

    additional_results.append({
        "classifier": type(case["classifier"]).__name__,
        "input": input_cols_str,
        "label": case["label"],
        "correct": correct,
        "within_one": within_one,
        "within_two": within_two,
        "avg_prob": avg_prob
    })

additional_results_df = spark.createDataFrame(additional_results)
additional_results_df = additional_results_df.withColumn("correct", F.round(F.col("correct"), 2))
additional_results_df = additional_results_df.withColumn("within_one", F.round(F.col("within_one"), 2))
additional_results_df = additional_results_df.withColumn("within_two", F.round(F.col("within_two"), 2))
additional_results_df = additional_results_df.withColumn("avg_prob", F.round(F.col("avg_prob"), 4))

display(additional_results_df)