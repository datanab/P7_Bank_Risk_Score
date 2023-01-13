from application import app
from flask import render_template, request, Flask, send_from_directory ,jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import json
import plotly
import plotly.graph_objects as go
import plotly.express as px
import requests
import os

# @app.route('/')
# def index():
#     return render_template('index.html')

def age_class(age):
    age_class = "less than 20 y.o."
    if (age>=20)&(age<30):
        age_class = "20 - 30 y.o."
    elif (age>=30)&(age<40):
        age_class = "30 - 40 y.o."
    elif (age>=40)&(age<50):
        age_class = "40 - 50 y.o."
    elif  age>=50:
        age_class = "more than 50 y.o."
    return age_class

def cpr_dict(df):
    aggregations = {}
    cpr_table_features = ['DAYS_BIRTH', 'CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    for feat in cpr_table_features:
        if feat in ['DAYS_BIRTH', 'CNT_CHILDREN']:
            aggregations[feat] = ["max","min","mean"]
        else:
            aggregations[feat] = pd.Series.mode
    b = df.groupby(["OCCUPATION_TYPE","NAME_FAMILY_STATUS","CLASS_AGE"]).agg(aggregations).reset_index()
    b["KEY"] = b[("OCCUPATION_TYPE","")]+"_"+b[("NAME_FAMILY_STATUS","")]+"_"+b[("CLASS_AGE","")]
    b = b.drop(columns = [("OCCUPATION_TYPE",""),("NAME_FAMILY_STATUS",""),("CLASS_AGE","")])
    b = b.set_index("KEY")
    new_col = []
    cpr_table_features = ['DAYS_BIRTH', 'CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    for feat in cpr_table_features:
        if feat in ['DAYS_BIRTH', 'CNT_CHILDREN']:
            new_col.append(feat+"_max")
            new_col.append(feat+"_min")
            new_col.append(feat+"_mean")
        else:
            new_col.append(feat+"_mode")
    new_col
    b.columns = new_col
    # Tous Fam Age
    c = df.groupby(["NAME_FAMILY_STATUS","CLASS_AGE"]).agg(aggregations).reset_index()
    c["KEY"] = "Tous"+"_"+c[("NAME_FAMILY_STATUS","")]+"_"+c[("CLASS_AGE","")]
    c = c.drop(columns = [("NAME_FAMILY_STATUS",""),("CLASS_AGE","")])
    c = c.set_index("KEY")
    c.columns = new_col
    # Occ Tous Age
    d = df.groupby(["OCCUPATION_TYPE","CLASS_AGE"]).agg(aggregations).reset_index()
    d["KEY"] = d[("OCCUPATION_TYPE","")]+"_"+"Tous"+"_"+d[("CLASS_AGE","")]
    d = d.drop(columns = [("OCCUPATION_TYPE",""),("CLASS_AGE","")])
    d = d.set_index("KEY")
    d.columns = new_col
    # Occ Fam Tous
    e = df.groupby(["OCCUPATION_TYPE","NAME_FAMILY_STATUS"]).agg(aggregations).reset_index()
    e["KEY"] = e[("OCCUPATION_TYPE","")]+"_"+e[("NAME_FAMILY_STATUS","")]+"_"+"Tous"
    e = e.drop(columns = [("OCCUPATION_TYPE",""),("NAME_FAMILY_STATUS","")])
    e = e.set_index("KEY")
    e.columns = new_col
    #Occ Tous Tous
    f = df.groupby(["OCCUPATION_TYPE"]).agg(aggregations).reset_index()
    f["KEY"] = f[("OCCUPATION_TYPE","")]+"_"+"Tous"+"_"+"Tous"
    f = f.drop(columns = [("OCCUPATION_TYPE","")])
    f = f.set_index("KEY")
    f.columns = new_col
    #Tous Fam Tous
    g = df.groupby(["NAME_FAMILY_STATUS"]).agg(aggregations).reset_index()
    g["KEY"] = "Tous"+ "_" +g[("NAME_FAMILY_STATUS","")] + "_" + "Tous"
    g = g.drop(columns = [("NAME_FAMILY_STATUS","")])
    g = g.set_index("KEY")
    g.columns = new_col
    #Tous Tous Age
    h = df.groupby(["CLASS_AGE"]).agg(aggregations).reset_index()
    h["KEY"] = "Tous"+ "_"  + "Tous" + "_" + h[("CLASS_AGE","")]
    h = h.drop(columns = [("CLASS_AGE","")])
    h = h.set_index("KEY")
    h.columns = new_col
    bcdefgh = pd.concat([b, c, d, e, f, g, h], axis = 0)
    l = {}
    for k in bcdefgh.index:
        for c in bcdefgh.columns:
            val = bcdefgh.at[k,c]
            if type(val)==np.float64:
                val_r = round(val,3)
                val = str(val_r)
            l[k + "_" + c] = str(val)
    return l

def formated_shapley_values(client_id):
    response = requests.post('http://datanab.pythonanywhere.com/shap/'+str(client_id)).content
    response_clean = str(response).replace("[","").replace("]","").replace(" ","").replace("b'","").replace("'","").split(",")
    shapley_v = np.array([np.array([float(x) for x in response_clean])])
    return shapley_v

def format_api_response(response_1):
    response_1_splited = str(response_1).split("#")
    pb = response_1_splited[0].replace("b'","")
    test = response_1_splited[1]
    feat_name = response_1_splited[2]
    test_fmt = test.replace("[","").replace("]","").replace(" ","").replace("'","").split(",")
    test_fmt_float = [float(x) for x in test_fmt]
    feat_name_fmt = feat_name.replace("[","").replace("]","").replace(" ","").replace("'","").replace("\"","").split(",")
    return pb, test_fmt_float, feat_name_fmt

def _force_plot_html(explainer, shap_values, ind):
        force_plot = shap.plots.force(shap_values[ind], matplotlib=False)
        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        return shap_html


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods = ["GET", "POST"])
def index():
    calcul = False
    client = 332040
    clients = pd.read_csv("application/static/data/clients_compare.csv")
    clients["DAYS_BIRTH"] = clients["DAYS_BIRTH"].apply(lambda age: np.abs(age)//365)
    clients["CLASS_AGE"] = clients["DAYS_BIRTH"].apply(lambda age: age_class(age))
    cpr_d = cpr_dict(clients)
    clients = clients.set_index("SK_ID_CURR")
    new_clients_index = np.loadtxt('application/static/data/new_clients_index.csv', delimiter=',')
    new_clients = clients.loc[new_clients_index,:]
    if request.method == "POST":
        client = request.form["client"]
    response_1 = requests.post('http://datanab.pythonanywhere.com/predict/'+str(client)).content
    print(response_1)
    pb, test_fmt_float, feat_name_fmt = format_api_response(response_1)
    prob = str(pb).replace("b\"","")
    features_names = pd.read_csv("application/static/data/features_names.csv")["features"]
    proba_float = 100*round(1-float(prob),3)
    classe = "1" if proba_float<50 else "0"
    class_mapping = {"0":"good_client", "1":"bad_client"}
    response_1_json = {"class" : class_mapping[classe], "probability" : proba_float}
    shapley = formated_shapley_values(int(client))
    if calcul:
        exp_val = requests.post('http://datanab.pythonanywhere.com/expected_value/'+str(client)).content
    else:
        exp_val = "0.631"
    shap.initjs()
    force = shap.force_plot(float(exp_val), shapley, test_fmt_float, feature_names=features_names, show = False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force.html()}</body>"
    if (proba_float > 0)&(proba_float <= 33):
        progress_color = "red"
    elif (proba_float > 33)&(proba_float <= 66):
        progress_color = "orange"
    elif (proba_float > 66)&(proba_float <= 100):
        progress_color = "green"
    # Graph score
    fig1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = proba_float,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Bank Score", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "white"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 19.5], 'color': '#da4c2b'},
            {'range': [20.5, 39.5], 'color': '#dd8435'},
            {'range': [40.5, 59.5], 'color': '#dfa33c'},
            {'range': [60.5, 79.5], 'color': '#bccf4e'},
            {'range': [80.5, 100.5], 'color': '#69cd4b'}],
        'threshold': {
            'line': {'color': "black", 'width': 1},
            'thickness': 0.75,
            'value': 50}
            }))
    fig1.update_layout(font = {'color': "black", 'family': "Arial"})
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


    clients_compare = pd.DataFrame(data = [["AMT_INCOME_TOTAL", clients["AMT_INCOME_TOTAL"].mean(),clients["AMT_INCOME_TOTAL"].max(),clients["AMT_INCOME_TOTAL"].min()],
                                            ["AMT_CREDIT",clients["AMT_CREDIT"].mean(),clients["AMT_CREDIT"].max(),clients["AMT_CREDIT"].min()],
                                            ["AMT_ANNUITY",clients["AMT_ANNUITY"].mean(),clients["AMT_ANNUITY"].max(),clients["AMT_ANNUITY"].min()]],
                                    columns = ["CREDIT INFORMATION","mean","max","min"]) 
    selected_client = pd.DataFrame(data = [["AMT_INCOME_TOTAL",clients.at[int(client),"AMT_INCOME_TOTAL"]],
                                        ["AMT_CREDIT",clients.at[int(client),"AMT_CREDIT"]],
                                        ["AMT_ANNUITY",clients.at[int(client),"AMT_ANNUITY"]]],
                                columns = ["CREDIT INFORMATION","values"])
    
    vs_income = pd.DataFrame(data = [["client", clients.at[int(client),"AMT_INCOME_TOTAL"],clients.at[int(client),"AMT_INCOME_TOTAL"],clients.at[int(client),"AMT_INCOME_TOTAL"]],
                                   ["global", clients["AMT_INCOME_TOTAL"].mean(),clients["AMT_INCOME_TOTAL"].min(),clients["AMT_INCOME_TOTAL"].max()]],
                                    columns = ["AMT_INCOME_TOTAL","mean","min","max"])

    vs_credit = pd.DataFrame(data = [["client", clients.at[int(client),"AMT_CREDIT"],clients.at[int(client),"AMT_CREDIT"],clients.at[int(client),"AMT_CREDIT"]],
                                   ["global", clients["AMT_CREDIT"].mean(),clients["AMT_CREDIT"].min(),clients["AMT_CREDIT"].max()]],
                                    columns = ["AMT_CREDIT","mean","min","max"])
    
    vs_annuity = pd.DataFrame(data = [["client", clients.at[int(client),"AMT_ANNUITY"],clients.at[int(client),"AMT_ANNUITY"],clients.at[int(client),"AMT_ANNUITY"]],
                                   ["global", clients["AMT_ANNUITY"].mean(),clients["AMT_ANNUITY"].min(),clients["AMT_ANNUITY"].max()]],
                                    columns = ["AMT_ANNUITY","mean","min","max"])
    
    vs = {"AMT_INCOME_TOTAL":vs_income, "AMT_CREDIT":vs_credit, "AMT_ANNUITY":vs_annuity}

    fig2_mean = px.bar(clients_compare, x='mean', y='CREDIT INFORMATION', color = ["blue","red","green"])
    fig2_mean.update_layout(yaxis={'visible': True, 'showticklabels':  True}, plot_bgcolor="rgb(255,255,255,0)", showlegend=False)
    graph2meanJSON = json.dumps(fig2_mean, cls=plotly.utils.PlotlyJSONEncoder)

    fig2_min = px.bar(clients_compare, x='min', y='CREDIT INFORMATION', color = ["blue","red","green"])
    fig2_min.update_layout(yaxis={'visible': True, 'showticklabels':  True}, plot_bgcolor="rgb(255,255,255,0)", showlegend=False)
    graph2minJSON = json.dumps(fig2_min, cls=plotly.utils.PlotlyJSONEncoder)

    fig2_max = px.bar(clients_compare, x='max', y='CREDIT INFORMATION', color = ["blue","red","green"])
    fig2_max.update_layout(yaxis={'visible': True, 'showticklabels':  True}, plot_bgcolor="rgb(255,255,255,0)", showlegend=False)
    graph2maxJSON = json.dumps(fig2_max, cls=plotly.utils.PlotlyJSONEncoder)

    fig3 = px.bar(selected_client, x='values', y='CREDIT INFORMATION', color = ["blue","red","green"])
    fig3.update_layout(yaxis={'visible': True, 'showticklabels':  True}, plot_bgcolor="rgb(255,255,255,0)", showlegend=False)
    # clients_no_outliers = clients[(clients["AMT_INCOME_TOTAL"]<1e7)&(clients["AMT_CREDIT"]<1e7)&(clients["AMT_ANNUITY"]<1e7)]
    # fig3.update_layout(xaxis_range=[0,clients_no_outliers[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY"]].max().max()])
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    vs_color = {"AMT_INCOME_TOTAL":["#90e0ef","#0077b6"], "AMT_CREDIT":["#e95555","#de1616"], "AMT_ANNUITY":["#6ede8a","#208b3a"]}

    figures = {}
    for feat in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
        for func in ["min","max","mean"]:
            f = px.bar(vs[feat], x=func, y=feat)
            f.update_layout(yaxis={'visible': True, 'showticklabels':  True}, plot_bgcolor="rgb(255,255,255,0)", showlegend=False)
            f.update_traces(marker_color=[vs_color[feat][0], vs_color[feat][1]])
            g = json.dumps(f, cls=plotly.utils.PlotlyJSONEncoder)
            figures[feat+"_"+func] = g
    

    liste_table_features = {"Gender":"CODE_GENDER",
                            "Age":"DAYS_BIRTH",
                            "Occupation":"OCCUPATION_TYPE",
                            "Housing type":"NAME_HOUSING_TYPE",
                            "Family status":"NAME_FAMILY_STATUS",
                            "Number of children":"CNT_CHILDREN",
                            "Vehicle":"FLAG_OWN_CAR",
                            "Realty":"FLAG_OWN_REALTY"}
    
    comparaison_table = {}
    for key in liste_table_features.keys():
        if key in ["Gender","Vehicle","Realty","Housing type","Occupation","Family status"]:
            comparaison_table[key] = {"min":"#","mean":"#","max":"#","mode":str(clients[liste_table_features[key]].mode().values[0]),"id":key.split()[0] + "_"}
        elif key in ["Age","Number of children"]:
            comparaison_table[key] = {"min":str(round(clients[liste_table_features[key]].min(),3)),
                                    "mean":str(round(clients[liste_table_features[key]].mean(),3)),
                                    "max":str(round(clients[liste_table_features[key]].max(),3)),
                                    "mode":"#",
                                    "id":key.split()[0]+"_"}
    
    cpr_table_features = ['DAYS_BIRTH', 'CODE_GENDER', 'FLAG_OWN_CAR','FLAG_OWN_REALTY', 'CNT_CHILDREN', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    tous_tous_tous = {}
    for feat in cpr_table_features:
        if feat in ['DAYS_BIRTH', 'CNT_CHILDREN']:
            tous_tous_tous["Tous_Tous_Tous_" + feat + '_min'] = str(clients[feat].min())
            tous_tous_tous["Tous_Tous_Tous_" + feat + '_mean'] = str(round(clients[feat].mean(),3))
            tous_tous_tous["Tous_Tous_Tous_" + feat + '_max'] = str(clients[feat].max())
        else:
            tous_tous_tous["Tous_Tous_Tous_" + feat + "_mode"] = clients[feat].mode()[0]

    plop = {"Gender":str(new_clients.at[int(client),"CODE_GENDER"]),
            "Age":str(new_clients.at[int(client),"DAYS_BIRTH"]),
            "Occupation":str(new_clients.at[int(client),"OCCUPATION_TYPE"]),
            "Housing type":str(new_clients.at[int(client),"NAME_HOUSING_TYPE"]),
            "Family status":str(new_clients.at[int(client),"NAME_FAMILY_STATUS"]),
            "Number of children":str(new_clients.at[int(client),"CNT_CHILDREN"]),
            "Vehicle":str(new_clients.at[int(client),"FLAG_OWN_CAR"]),
            "Realty":str(new_clients.at[int(client),"FLAG_OWN_REALTY"])}
    ids = {"Gender":{"min":"Gender_min","mean":"Gender_mean","max":"Gender_max","mode":"Gender_mode"},
            "Age":{"min":"Age_min","mean":"Age_mean","max":"Age_max","mode":"Age_mode"},
            "Occupation":{"min":"Occupation_min","mean":"Occupation_mean","max":"Occupation_max","mode":"Occupation_mode"},
            "Housing type":{"min":"Housing_min","mean":"Housing_mean","max":"Housing_max","mode":"Housing_mode"},
            "Family status":{"min":"Family_min","mean":"Family_mean","max":"Family_max","mode":"Family_mode"},
            "Number of children":{"min":"Number_min","mean":"Number_mean","max":"Number_max","mode":"Number_mode"},
            "Vehicle":{"min":"Vehicle_min","mean":"Vehicle_mean","max":"Vehicle_max","mode":"Vehicle_mode"},
            "Realty":{"min":"Realty_min","mean":"Realty_mean","max":"Realty_max","mode":"Realty_mode"}}
    plop_cpr = {"Gender":str(clients["CODE_GENDER"].mode().values[0]),
            "Age":str(clients["DAYS_BIRTH"].mean()),
            "Occupation":str(clients["OCCUPATION_TYPE"].mode().values[0]),
            "Housing type":str(clients["NAME_HOUSING_TYPE"].mode().values[0]),
            "Family status":str(clients["NAME_FAMILY_STATUS"].mode().values[0]),
            "Number of children":str(int(clients["CNT_CHILDREN"].mode().values[0])),
            "Vehicle":str(clients["FLAG_OWN_CAR"].mode().values[0]),
            "Realty":str(clients["FLAG_OWN_REALTY"].mode().values[0])}
    
    liste_occupation = list(clients["OCCUPATION_TYPE"].dropna().unique())
    liste_family_status = list(clients["NAME_FAMILY_STATUS"].dropna().unique())
    liste_age_class = list((clients.sort_values(by=["DAYS_BIRTH"]))["CLASS_AGE"].dropna().unique())
    final_cpr = {**cpr_d, **tous_tous_tous}
    d = json.dumps(final_cpr)
    return render_template('layout.html',
                            graph1JSON=graph1JSON,
                            graph2minJSON=graph2minJSON,
                            graph2maxJSON=graph2maxJSON,
                            graph2meanJSON=graph2meanJSON,
                            graph3JSON=graph3JSON,
                            income_min = figures["AMT_INCOME_TOTAL_min"],
                            income_max = figures["AMT_INCOME_TOTAL_max"],
                            income_mean = figures["AMT_INCOME_TOTAL_mean"],
                            credit_min = figures["AMT_CREDIT_min"],
                            credit_max = figures["AMT_CREDIT_max"],
                            credit_mean = figures["AMT_CREDIT_mean"],
                            annuity_min = figures["AMT_ANNUITY_min"],
                            annuity_max = figures["AMT_ANNUITY_max"],
                            annuity_mean = figures["AMT_ANNUITY_mean"],
                            response = response_1_json,
                            client_id = client,
                            plop = plop,
                            plop_cpr = plop_cpr,
                            comparaison_table = comparaison_table,
                            liste_occupation = liste_occupation,
                            liste_family_status = liste_family_status,
                            liste_age_class = liste_age_class,
                            ids = ids,
                            drapo = d,
                            shaplot = shap_html)


