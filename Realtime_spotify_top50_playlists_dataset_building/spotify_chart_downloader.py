import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

from bs4 import BeautifulSoup
import requests, re, types

import pandas as pd
from datetime import date, datetime




class spotify_daily_charts_downloader:
    
    def __init__(self, username, scope, client_id, client_secret):
        
        self.username = username
        self.scope = scope
        self.client_id = client_id
        self.client_secret = client_secret
        self.spotify = None
        
        self.available_countries = None
        self.chart_ids = None
        self._columns = ['continent', 'country', 'rank', 'song', 'artist', 'album', 'release_date', 'song_popularity', 'song_id', 'artist_id', 'album_id', 'date']
        self.global_chart = pd.DataFrame(columns = self._columns)
        
    def authorization(self):
        
        def playlist_track(self, playlist_id):
            """ 
            returns a single artist given the artist's ID, URI or URL

            Input: a list of playlist_ids [id, id, id...]
            output: json format track description
            """
            return self._get('playlists/' + playlist_id + '/tracks')
        
        client_credentials_manager = SpotifyClientCredentials(client_id = self.client_id, client_secret = self.client_secret)
        self.spotify = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
        self.spotify.playlist_track = types.MethodType(playlist_track, self.spotify)
        
    def download_chart(self):
        
        self.authorization()
        self.get_available_countries()
        self.collect_chart_ids()
        self.merge_tracks_information()
        self.merge_songs_information()
        self.merge_artists_information()
        self.merge_albums_information()
        self.reorder_columns()
        self.download_dataframe()
        
    def get_available_countries(self):
        """
        Crawling names of countries that are available to use Spotify.
        Name of countries are nested in the Name of Continent.

        Input: None
        Output: A Dictionary of Continent:Country names
                {continent : [country, country, ...]}
        """
        url = "https://support.spotify.com/us/using_spotify/the_basics/full-list-of-territories-where-spotify-is-available/"
        html = requests.get(url).text

        soup = BeautifulSoup(html, 'html.parser')

        # available_countries[continent] = [country, country, country]
        available_countries = {}
        target_tags = soup.find_all(['td'])
        for tag in target_tags:
            countries = re.findall('[\w\s]+(?=[.,])', tag.text)
            if countries:
                countries = [country.lstrip() for country in countries]
                available_countries[continent] = countries
            else:
                continent = tag.text

        self.available_countries = available_countries
    
    def collect_chart_ids(self):
        """
        Collect playlist ids through Spotify API

        Input: A Dictionary of {continent: [country, country...], ....}
        Output: A tuple of ((continent, country, playlist_id), ...)
        """
        chart_ids = []
        for continent in self.available_countries.keys():
            countries = self.available_countries[continent]

            for country in countries:
                playlist_name = country + ' Top 50'
                playlists = self.spotify.search(q = playlist_name, type = 'playlist')
                for playlist in playlists['playlists']['items']:

                    # If not Spotify Official playlist
                    if playlist['owner']['display_name'] != 'spotifycharts':
                        continue
                    # If not Top 50 ex) Viral 50
                    if playlist['name'] != playlist_name:
                        continue

                    chart_ids.append((continent, country, playlist['id']))

        self.chart_ids = chart_ids
    
    def collect_chart(self, continent, country, tracks):
        """
        Picking out only required information from track information

        Input: continent(String), country(String), tracks(json)
        Output: A list of dictionary that contains track information
                [{'continent':, 'country': , 'rank': , 'song': , 'artist': , ...}, ]
        """
        track_infos = []
        for idx, track in enumerate(tracks['items'], 1):
            track_info = {'continent': '', 'country' : '', 'rank': 0, 'song' : '', 'artist' : '', 'album' : '', 'song_id': '', 'artist_id': '', 'album_id': '', 'release_data': '', 'song_popularity' : 0, 'date': ''}

            track_info['rank'] = idx

            track_info['continent'] = continent
            track_info['country'] = country

            track_info['song'] = track['track']['name']
            track_info['song_id'] = track['track']['id']
            track_info['song_popularity'] = track['track']['popularity']

            track_info['album'] = track['track']['album']['name']
            track_info['album_id'] = track['track']['album']['id']

            try:
                track_info['release_date'] = datetime.strptime(track['track']['album']['release_date'], '%Y-%m-%d')
            except:
                track_info['release_date'] = datetime.strptime(track['track']['album']['release_date'], '%Y')

            track_info['artist'] = track['track']['artists'][0]['name']
            track_info['artist_id'] = track['track']['artists'][0]['id']

            track_info['date'] = date.today()
            track_infos.append(track_info)

        return track_infos
        
    def merge_tracks_information(self):

        """
        for loop below COLLECTs tracks information of each country and
        MERGEs it into global_chart dataframe
        """

        for (continent, country, playlist_id) in self.chart_ids:

            tracks = self.spotify.playlist_track(playlist_id)
            dict_country_chart = self.collect_chart(continent, country, tracks)

            country_chart = pd.DataFrame(dict_country_chart, columns = self._columns)
            self.global_chart = pd.concat([self.global_chart, country_chart], ignore_index = True)
            
    def return_ids(self, requested_type, limits):
        """
        A generator that returns requested data(ids) in 50 pieces (Spotify API limits 50 ids per request)

        Input: requested_type(string), name of columns  ex) 'song_id', 'artist_id', 'album_id'
               limits(input), number of items returns each time  ex) 50, 20 
        Output: A list of collected ids ['id', 'id', 'id']
        """
        ids = []
        for idx, song_id in enumerate((x for x in self.global_chart[requested_type].tolist()), 1):
            ids.append(song_id)
            if idx % limits == 0:
                yield ids
                ids.clear()

        if len(ids):
            yield ids

    def merge_songs_information(self):
        _columns = ['song_id', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence']
        songs_df = pd.DataFrame(columns = _columns)

        """
        for loop below COLLECTs songs information 
        MERGEs it into songs_df dataframe
        """

        for x in self.return_ids('song_id', 50):

            songs_features = []
            song_infos = self.spotify.audio_features(x)

            for idx, song_info in enumerate(song_infos):
                #print(idx,song_info)
                song_features = {'song_id': '', 'acousticness': None, 'danceability': None, 'energy': None, 'instrumentalness': None, 'liveness': None, 'loudness': None, 'speechiness': None, 'valence': None}

                try:        
                    song_features['song_id'] = song_info['id']
                    song_features['acousticness'] = song_info['acousticness']
                    song_features['danceability'] = song_info['danceability']
                    song_features['energy'] = song_info['energy']
                    song_features['instrumentalness'] = song_info['instrumentalness']
                    song_features['liveness'] = song_info['liveness']
                    song_features['loudness'] = song_info['loudness']
                    song_features['speechiness'] = song_info['speechiness']
                    song_features['valence'] = song_info['valence']
                except:
                    """
                    Some of songs do not have analysis data yet, so put Nan first.
                    But keeping song_id.

                    Checked no feature analysis data from Spotify API
                    """
                    song_features['song_id'] = x[idx]

                songs_features.append(song_features)

            temp_songs_df = pd.DataFrame(songs_features, columns = _columns)
            songs_df = pd.concat([songs_df, temp_songs_df], ignore_index = True)
        
        self.global_chart = pd.concat([self.global_chart, songs_df], axis = 1)
        self.global_chart = self.global_chart.loc[:,~self.global_chart.columns.duplicated()]
    
    def merge_artists_information(self):
        
        _columns = ['artist_id', 'genre', 'artist_popularity', 'followers']
        artists_df = pd.DataFrame(columns = _columns)

        for x in self.return_ids('artist_id', 50):

            artists_features = []

            artist_infos = self.spotify.artists(x)

            for artist_info in artist_infos['artists']:
                artist_features = {'artist_id': '', 'genre': '', 'artist_popularity': 0, 'followers': 0}
                artist_features['artist_id'] = artist_info['id']
                try:
                    artist_features['genre'] = artist_info['genres'][0].split(' ')[0]
                except:
                    # TTD : Searching Genre from google? 
                    artist_features['genre'] = None
                    #print(artist_info['genres'], artist_info['name'])
                artist_features['artist_popularity'] = artist_info['popularity']
                artist_features['followers'] = artist_info['followers']['total']

                artists_features.append(artist_features)

            temp_artists_df = pd.DataFrame(artists_features, columns = _columns)
            artists_df = pd.concat([artists_df, temp_artists_df], ignore_index = True)
        
        self.global_chart = pd.concat([self.global_chart, artists_df], axis = 1)
        self.global_chart = self.global_chart.loc[:,~self.global_chart.columns.duplicated()]
        
    def merge_albums_information(self):
        
        _columns = ['album_id', 'album_popularity']
        albums_df = pd.DataFrame(columns = _columns)

        for x in self.return_ids('album_id', 20):

            albums_features = []
            albums_infos = self.spotify.albums(x)

            for album_info in albums_infos['albums']:
                album_features = {'album_id': '', 'album_popularity': 0}
                album_features['album_id'] = album_info['id']
                album_features['album_popularity'] = album_info['popularity']

                albums_features.append(album_features)

            temp_albums_df = pd.DataFrame(albums_features, columns = _columns)
            albums_df = pd.concat([albums_df, temp_albums_df], ignore_index = True)
        
        self.global_chart = pd.concat([self.global_chart, albums_df], axis = 1)
        self.global_chart = self.global_chart.loc[:,~self.global_chart.columns.duplicated()]
        
    def reorder_columns(self):
        
        cols = ['date','continent','country','rank', 'song','artist','album','genre','followers','song_popularity','artist_popularity','album_popularity','release_date','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','valence','song_id','artist_id','album_id']
        self.global_chart = self.global_chart[cols]
        
    def download_dataframe(self):
        
        self.global_chart.to_csv('spotify_top50_chart.csv')