from django.urls import path
from . import views
from .views import *



urlpatterns = [
    path('',views.home,name="home"),
    path('assignment1/',views.assignment1,name="assignment1"),
    path('assignment1_que2/',views.assignment1_que2,name="assignment1_que2"),
    path('assignment2/',views.assignment2,name="assignment2"),
    path('assignment3/',views.assignment3,name="assignment3"),
    path('assignment3/confuse_matrix/',views.assignment3_confuse_matrix,name="assignment3_confuse_matrix"),
    path('assignment4/',views.assignment4,name="assignment4"),
    path('assignment5/',views.assignment5,name="assignment5"),
    path('api/upload-csv/', CSVFileUploadView.as_view(), name='upload-csv-api'),
    path('assignment55/',RegressionClass.as_view(),name="regression"),
      path('dendrogram/', views.dendrogram_view, name='dendrogram'),
    path('pagerank/', views.get_page_rank, name='calculate_page_rank'),
    path('hitscore/', views.get_hits_scores, name='calculate_hits_scores'),
    path('crawl/', views.crawl_urls, name='crawl_urls'),
    path('kmeans/', views.kmeans_view, name='kmeans'),
    path('kmedoids/', views.kmedoids_view, name='kmedoids'),
    # write path for dbscan_view and birch_view
    path('birch/', views.birch_view, name='birch'),
    path('dbscan/', views.dbscan_view, name='dbscan'),
    # for clustering
    path('clustering/',views.clustering_view,name='clustering'),
    path('run_association_rules/', views.run_association_rules, name='run_association_rules'),
    



]