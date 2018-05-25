import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;

public class JavaRecommendationExample {
    public static class Similarity {
        public int Friend;
        public int Sim;

        public Similarity(int x, int y){
            Friend = x;
            Sim = y;
        }
    }
    public static void main(String[] args) throws IOException {
        int userID = 1029;
        SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example").setMaster("local[2]") ;
        JavaSparkContext jsc = new JavaSparkContext(conf);
        String pathTracks = "/home/cloudera/IdeaProjects/spark/datasetTracks.csv";
        String pathArtists = "/home/cloudera/IdeaProjects/spark/datasetArtists.csv";
        JavaPairRDD dataTracks = getSimilarUsers(userID, jsc, pathTracks);
        JavaPairRDD dataArtists = getSimilarUsers(userID, jsc, pathArtists);
    }

    private static JavaPairRDD getSimilarUsers(int userID, JavaSparkContext jsc, String pathArtists) {
        JavaRDD<String> data = jsc.textFile(pathArtists);
        JavaRDD<Rating> ratings = data.map(s -> {
            String[] sarray = s.split(",");
            return new Rating(Integer.parseInt(sarray[0]),
                    Integer.parseInt(sarray[1]),
                    Double.parseDouble(sarray[2]));
        });
        List<Rating> controlUserRatings = ratings.filter(s -> (s.user() == userID)).collect();
        HashSet<Integer> controlUserTracksId = new HashSet<>();
        for (Rating raiting : controlUserRatings) {
            controlUserTracksId.add(raiting.product());
        }
        Broadcast<HashSet> sharedHashSet = jsc.broadcast(controlUserTracksId);
        JavaRDD<Rating> similarUsers = ratings.filter(s -> sharedHashSet.value().contains(s.product()));

        JavaPairRDD ratingsByTracks = similarUsers
                .mapToPair(s -> new Tuple2(s.user(), s.rating()))
                .reduceByKey((a, b) -> new Similarity((int)a, (int) a + (int) b));
        return ratingsByTracks;
    }
}