import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;
import java.util.*;

public class JavaRecommendationExample {

    public static class UserTrackInfoRating implements Serializable{
        public int User;
        public int Track;
        public int Artist;
        public double Rating;

        public UserTrackInfoRating(int userId, int trackId, int artistId, double rating){
            User = userId;
            Track = trackId;
            Artist = artistId;
            Rating = rating;
        }
    }

    public static void main(String[] args)  {
        int userID = 1029;
        SparkConf conf = new SparkConf()
                .setAppName("Java Collaborative Filtering Example")
                .setMaster("local[2]") ;
        JavaSparkContext sparkContext = new JavaSparkContext(conf);
        String pathTracks = "/home/cloudera/IdeaProjects/spark/datasetTracks.csv";
        String pathArtists = "/home/cloudera/IdeaProjects/spark/datasetArtists.csv";

        //Read data to RDD (UserID, TrackID, ArtistID, TrackRating)
        JavaRDD<String> input = sparkContext.textFile(pathTracks);
        JavaRDD<UserTrackInfoRating> trackRatings = input
                .map(line -> {
                    String[] array = line.split(",");
                    return new UserTrackInfoRating(
                            Integer.parseInt(array[0]),
                            Integer.parseInt(array[1]),
                            Integer.parseInt(array[2]),
                            Double.parseDouble(array[3]));
                });

        //Fill control user tracks info to shared HashMap (TrackID -> Rating)
        List<UserTrackInfoRating> controlUserTracksRatings = trackRatings
                .filter(item -> (item.User == userID))
                .collect();
        HashMap<Integer, Double> controlUserTracksEvals = new HashMap<>();
        for (UserTrackInfoRating rating : controlUserTracksRatings) {
            controlUserTracksEvals.put(rating.Track, rating.Rating);
        }
        Broadcast<HashMap> sharedControlUserTracksEvalsHashMap = sparkContext
                .broadcast(controlUserTracksEvals);

        //Read data to format (UserID, ArtistID, ArtistRating)
        input = sparkContext.textFile(pathArtists);
        JavaRDD<Rating> artistsRatings = input.map(line -> {
            String[] array = line.split(",");
            return new Rating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Double.parseDouble(array[2]));
        });

        //Fill control user artists info to shared HashMap (ArtistID -> Rating)
        List<Rating> controlUserArtistsRatings = artistsRatings
                .filter(item -> (item.user() == userID))
                .collect();
        HashMap<Integer, Double> controlUserAtristsEvals = new HashMap<>();
        for (Rating rating : controlUserArtistsRatings) {
            controlUserAtristsEvals.put(rating.product(), rating.rating());
        }
        Broadcast<HashMap> sharedControlUserArtistsEvalsHashMap = sparkContext
                .broadcast(controlUserAtristsEvals);

        //Fill similar users tracks to shared HashMap (SimilarUserID -> List<TrackInfo>)
        JavaRDD<UserTrackInfoRating> similarUsersByTracks = trackRatings
                .filter(item -> (sharedControlUserTracksEvalsHashMap
                        .value()
                        .containsKey(item.Track) && item.User != userID));
        List<UserTrackInfoRating> similarUsersByTracksList = similarUsersByTracks
                .collect();
        HashMap<Integer, List<UserTrackInfoRating>> similarUserHashMap = new HashMap<>();
        for (UserTrackInfoRating trackInfo : similarUsersByTracksList) {
            if (similarUserHashMap.get(trackInfo.User) == null){
                similarUserHashMap.put(trackInfo.User, new ArrayList<>());
            }
            similarUserHashMap.get(trackInfo.User).add(trackInfo);
        }
        Broadcast<HashMap> sharedSimilarUserTracksHashMap = sparkContext
                .broadcast(similarUserHashMap);

        //Transformation RDD from (UserID, TrackID, ArtistID, TrackRating)
        //to (UserID, TrackID, TrackRating)
        JavaPairRDD<Integer, Tuple2> similarUsersTracksEvals = similarUsersByTracks
                .mapToPair(item -> new Tuple2(item.User, new Tuple2(item.Track, item.Rating)));

        //Group RDD by userID
        JavaPairRDD<Integer, Iterable<Tuple2>> similarUsersTrackGrouped = similarUsersTracksEvals
                .groupByKey();

        //Calculate Rating Of Similarity Users (RatingOfSimilarity, UserID)
        JavaPairRDD<Double, Integer> similarUserRatings = similarUsersTrackGrouped
                .mapToPair(item ->
                        calculateRatingOfSimilarityUsers(sharedControlUserArtistsEvalsHashMap, item));

        //Sort users by rating of similarity
        JavaPairRDD<Double, Integer> similarUserSorted = similarUserRatings
                .sortByKey(false);

        //For each user get sorted users tracks by rating of evaluation this track by control user
        JavaRDD<UserTrackInfoRating> similarUserTrackEval = similarUserSorted
                .flatMap(item ->
                    getSortedUserTracks(sharedControlUserArtistsEvalsHashMap, sharedSimilarUserTracksHashMap, item._2));

        //Prepare and out recommended tracks
        List<Integer> result = similarUserTrackEval.map(item -> item.Track).collect();
        result.forEach(item -> System.out.println(item));
    }

    private static Iterator<UserTrackInfoRating> getSortedUserTracks(
            Broadcast<HashMap> sharedControlUserArtistsEvalsHashMap,
            Broadcast<HashMap> sharedSimilarUserTracksHashMap,
            Integer userId) {
        //Read user List<TrackInfo> from shared hashMap
        ArrayList<UserTrackInfoRating> friendTracksInfo = (ArrayList<UserTrackInfoRating>)
                sharedSimilarUserTracksHashMap
                        .value()
                        .get(userId);

        //For each tracks calculate control user evaluation of this track
        friendTracksInfo.forEach(track -> {
            track.Rating = 0.0;
            if (sharedControlUserArtistsEvalsHashMap.value().containsKey(track.Artist)){
                track.Rating = Math.log((double) sharedControlUserArtistsEvalsHashMap.value().get(track.Artist));
            }
        });

        //Sort tracks by rating
        friendTracksInfo.sort(Collections.reverseOrder(Comparator.comparing(track -> track.Rating)));
        return friendTracksInfo.iterator();
    }

    private static Tuple2<Double, Integer> calculateRatingOfSimilarityUsers(
            Broadcast<HashMap> sharedControlUserArtistsEvalsHashMap,
            Tuple2<Integer, Iterable<Tuple2>> userTracks) {
        Iterator tracks = userTracks._2.iterator();
        double resultEval = 0.0;
        while (tracks.hasNext()){
            Tuple2 productEval = (Tuple2) tracks.next();
            double friendUserEval = (double) productEval._2;
            double controlUserEval = (double) sharedControlUserArtistsEvalsHashMap.value().get(productEval._1);
            resultEval += friendUserEval + controlUserEval - Math.abs(friendUserEval - controlUserEval);
        }
        return new Tuple2(resultEval, userTracks._1);
    }
}