import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;
import java.util.*;

public class JavaRecommendationExample {

    public static class CustomRating {
        public int User;
        public int Track;
        public int Artist;
        public double Rating;

        public CustomRating(int userId, int trackId, int artistId, double rating){
            User = userId;
            Track = trackId;
            Artist = artistId;
            Rating = rating;
        }
    }
    public static void main(String[] args)  {
        int userID = 1029;
        SparkConf conf = new SparkConf().setAppName("Java Collaborative Filtering Example").setMaster("local[2]") ;
        JavaSparkContext jsc = new JavaSparkContext(conf);
        String pathTracks = "/home/cloudera/IdeaProjects/spark/datasetTracks.csv";
        String pathArtists = "/home/cloudera/IdeaProjects/spark/datasetArtists.csv";

        JavaRDD<String> data = jsc.textFile(pathTracks);
        JavaRDD<CustomRating> trackRatings = data.map(s -> {
            String[] array = s.split(",");
            return new CustomRating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Integer.parseInt(array[2]),
                    Double.parseDouble(array[3]));
        });
        List<CustomRating> controlUserTracksRatings = trackRatings.filter(s -> (s.User == userID)).collect();
        HashMap<Integer, Double> controlUserTracksEvals = new HashMap<>();
        for (CustomRating rating : controlUserTracksRatings) {
            controlUserTracksEvals.put(rating.Track, rating.Rating);
        }
        Broadcast<HashMap> controlUserTracksEvalsHashMap = jsc.broadcast(controlUserTracksEvals);

        data = jsc.textFile(pathArtists);
        JavaRDD<Rating> artistsRatings = data.map(s -> {
            String[] array = s.split(",");
            return new Rating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Double.parseDouble(array[2]));
        });
        List<Rating> controlUserAtristsRatings = artistsRatings.filter(s -> (s.user() == userID)).collect();
        HashMap<Integer, Double> controlUserAtristsEvals = new HashMap<>();
        for (Rating rating : controlUserAtristsRatings) {
            controlUserAtristsEvals.put(rating.product(), rating.rating());
        }
        Broadcast<HashMap> controlUserArtistsEvalsHashMap = jsc.broadcast(controlUserAtristsEvals);




        JavaRDD<CustomRating> similarUsersByTracks = trackRatings
                .filter(s -> controlUserTracksEvalsHashMap.value().containsKey(s.Track));
        List<CustomRating> similarUsersByTracksList = similarUsersByTracks.collect();
        HashMap<Integer, List<CustomRating>> similarUserHashMap = new HashMap<>();
        for (CustomRating rating : similarUsersByTracksList) {
            if (similarUserHashMap.get(rating.User) == null){
                similarUserHashMap.put(rating.User, new ArrayList<>());
            }
            similarUserHashMap.get(rating.User).add(rating);
        }
        Broadcast<HashMap> sharedSimilarUserHashMap = jsc.broadcast(similarUserHashMap);

        JavaPairRDD<Integer, Tuple2> similarUsersTracksEvals = similarUsersByTracks
                .mapToPair(s -> new Tuple2(s.User, new Tuple2(s.Track, s.Rating)));
        JavaPairRDD<Integer, Iterable<Tuple2>> similarUsersTrackGroupped = similarUsersTracksEvals.groupByKey();
        JavaPairRDD similarUserRatings = similarUsersTrackGroupped.mapToPair(s -> {
            Iterator iter = s._2.iterator();
            double resultEval = 0.0;
            while (iter.hasNext()){
                Tuple2 productEval = (Tuple2) iter.next();
                double friendUserEval = (double) productEval._2;
                double controlUserEval = (double) controlUserTracksEvalsHashMap.value().get(productEval._1);
                resultEval += friendUserEval + controlUserEval - Math.abs(friendUserEval - controlUserEval);
            }
            return new Tuple2(resultEval, s._1);
        });
        JavaPairRDD<Double, Integer> similarUserSortedWithSwap = similarUserRatings.sortByKey();
        JavaRDD<Integer> similarUserSorted = similarUserSortedWithSwap.map(s -> s._2);
        JavaRDD<CustomRating> similarUserTrackEval = similarUserSorted
                .flatMap(s -> {
                    ArrayList<CustomRating> friendTracksInfo = (ArrayList<CustomRating>) sharedSimilarUserHashMap.value().get(s);
                    friendTracksInfo.forEach(e -> {
                        e.Rating = 0.0;
                        if (controlUserArtistsEvalsHashMap.value().containsKey(e.Artist)){
                            e.Rating = Math.log((double) controlUserArtistsEvalsHashMap.value().get(e.Artist));
                        }
                    });
                    friendTracksInfo.sort(Comparator.comparing(x -> x.Rating));
                    return friendTracksInfo.iterator();
                });
        List<Integer> result = similarUserTrackEval.map(r -> r.Track).collect();
        result.forEach(i -> System.out.println(i));
    }
}