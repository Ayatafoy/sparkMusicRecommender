import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;

import java.io.IOException;
import java.util.*;

public class JavaRecommendationExample {

    public static class Similarity {
        public int Friend;
        public double Sim;

        public Similarity(int x, double y){
            Friend = x;
            Sim = y;
        }
    }
    public static void main(String[] args)  {
        int userID = 1029;
        SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example").setMaster("local[2]") ;
        JavaSparkContext jsc = new JavaSparkContext(conf);
        String pathTracks = "/home/cloudera/IdeaProjects/spark/datasetTracks.csv";
        String pathArtists = "/home/cloudera/IdeaProjects/spark/datasetArtists.csv";

        JavaRDD<String> data = jsc.textFile(pathTracks);
        JavaRDD<Rating> ratings = data.map(s -> {
            String[] array = s.split(",");
            return new Rating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Double.parseDouble(array[2]));
        });
        List<Rating> controlUserRatings = ratings.filter(s -> (s.user() == userID)).collect();
        HashMap<Integer, Double> controlUserTracksEvals = new HashMap<>();
        for (Rating rating : controlUserRatings) {
            controlUserTracksEvals.put(rating.product(), rating.rating());
        }
        Broadcast<HashMap> sharedHashMap = jsc.broadcast(controlUserTracksEvals);
        JavaRDD<Rating> similarUsersByTracks = ratings.filter(s -> sharedHashMap.value().containsKey(s.product()));
        JavaPairRDD<Integer, Tuple2> similarUsersTracksEvals = similarUsersByTracks.mapToPair(s -> new Tuple2(s.user(), new Tuple2(s.product(), s.rating())));
        JavaPairRDD<Integer, Iterable<Tuple2>> similarUsersTrackGroupped = similarUsersTracksEvals.groupByKey();
        JavaPairRDD similarUserRatings = similarUsersTrackGroupped.mapToPair(s -> {
            Iterator iter = s._2.iterator();
            double resultEval = 0.0;
            while (iter.hasNext()){
                Tuple2 productEval = (Tuple2) iter.next();
                double friendUserEval = (double) productEval._2;
                double controlUserEval = (double) sharedHashMap.value().get(productEval._1);
                resultEval += friendUserEval + controlUserEval - Math.abs(friendUserEval - controlUserEval);
            }
            return new Tuple2(resultEval, s._1);
        });
        JavaPairRDD<Double, Integer> similarUserSortedWithSwap = similarUserRatings.sortByKey();
        List<Integer> similarUserSorted = similarUserSortedWithSwap.map(s -> s._2).collect();
    }

    private static JavaRDD<Rating> getSimilarUsers(int userID, JavaSparkContext jsc, String pathArtists) {
        JavaRDD<String> data = jsc.textFile(pathArtists);
        JavaRDD<Rating> ratings = data.map(s -> {
            String[] array = s.split(",");
            return new Rating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Double.parseDouble(array[2]));
        });
        List<Rating> controlUserRatings = ratings.filter(s -> (s.user() == userID)).collect();
        HashSet<Integer> controlUserTracksId = new HashSet<>();
        for (Rating rating : controlUserRatings) {
            controlUserTracksId.add(rating.product());
        }
        Broadcast<HashSet> sharedHashSet = jsc.broadcast(controlUserTracksId);
        JavaRDD<Rating> similarUsers = ratings.filter(s -> sharedHashSet.value().contains(s.product()));
        return similarUsers;
    }
}