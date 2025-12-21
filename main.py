# main.py
from src.data_loader import load_supply_chain_data
from src.preprocessing import clean_data
from src.feature_engineering import add_subclassification_risk, add_risk_score, create_delay_target, create_time_features, finalize_features,save_processed_data, scale_numeric_features,add_joint_risk_score,drop_features
from src.visualization import visualize_delay_counts, visualize_shipment_mode, visualize_cost_vs_delay,visualize_delay_by_shipment_mode,visualize_delay_proportion_by_shipment_mode,visualize_top_countries_delay_by_shipment_mode, delay_rate_by_weight_and_mode,delay_percentage_by_manufacturing_site,delay_percentage_by_fulfill_via,visualize_vendor_fulfill_delay_percentage
from utils.Analyze import delay_proportion_by_subclassification,check_delay_counts_by_mode_and_country,count_vendors_by_fulfill_via,vendor_fulfill_via_delay_counts,analyze_delay_by_weight,missing_values_percentage_in_PQ_to_PO_days

def run_pipeline():
    # 1. Load & Clean
    df = load_supply_chain_data("C:\\Users\\jesme\\OneDrive\\Desktop\\Supply_chain_ml_project\\data\\Raw\\SCMS_Delivery_History_Dataset_20150929.csv")
    df = clean_data(df)

    # 2. Feature Engineering
    df = create_delay_target(df)
    df = create_time_features(df)
   # 3. Visualizations
    visualize_delay_counts(df)
    visualize_shipment_mode(df)
    # visualize_cost_vs_delay(df)
    visualize_delay_by_shipment_mode(df)
    #Here we got the real view of data that ocean mode has the highest proportion of delays
    visualize_delay_proportion_by_shipment_mode(df)

    visualize_top_countries_delay_by_shipment_mode(df, top_n=15)

    delay_percentage_by_fulfill_via(df)
    visualize_vendor_fulfill_delay_percentage(df)

    # Additional Analysis

    #Through this function we got to know that only 1 vendor is using rdd as fulfill via method and that heps us to prevent overfittig as if we only consider full via than there is 17.15 percent delay rate but in reality its only one vendor causing this high delay rate
    count_vendors_by_fulfill_via(df)

    #So this shows fulfill via method is not making any meaningfull impact on our data and also this is a classic issue of confounding variable where vendor and fulfill via are confounded. Hence we should drop fulfill via column
    vendor_fulfill_via_delay_counts(df)
    #dropping fulfill via column as it is not making any meaningfull impact on delay prediction and also it is confounded with vendor
    df = drop_features(df, columns=["Fulfill Via"])

    #checking missing values percentage in PQ_to_PO_days column
    missing_values_percentage_in_PQ_to_PO_days(df)
    #we can drop this column as it has more than 60 percent missing values
    df = drop_features(df, columns=["PQ_to_PO_days"])
    delay_rate_by_weight_and_mode(df) #this shows that weight doesnt making any impact on delay rate Hence weight and shipment mode are not correlated, Delay is not strongly driven by weight + shipment mode together

    # Does the manufacturing site (Manufacturer / Country of Origin) affect delay percentage?
    #analysis shows that a small subset of manufacturers consistently exhibits higher delay percentages compared to others. This suggests manufacturer-specific operational or logistical factors may contribute to delivery risk. These insights can be encoded as manufacturer-level risk features for predictive modeling
    delay_percentage_by_manufacturing_site(df)

    analyze_delay_by_weight(df, q=4)
    
    print("------------------------------------")
    delay_proportion_by_subclassification(df)
    print("------------------------------------")
    # Risk scores
    #these two features are correlated for delay so we can create a composite risk score based on these features [shipment mode and country]
    # df = add_risk_score(df, "Shipment Mode")
    # df = add_risk_score(df, "Country")
    #To handle multicollinearity we can combine these two features into a single composite risk score and it would tell us that historically, how risky is this shipment mode when used in this country?
    df = add_joint_risk_score(df, "Shipment Mode", "Country")
    df = add_risk_score(df, "Manufacturing Site")

        # Subclassification risk
    df = add_subclassification_risk(df)
    df = drop_features(df, columns=["Sub Classification"])
   

    # Finalize
    df = finalize_features(df)
    df = scale_numeric_features(df)


    # Save cleaned CSV
    # df.to_csv("data/processed/supply_chain_clean.csv", index=False)
    
    save_processed_data(df, r"data/processed/supply_chain_final.csv")






    df_new = load_supply_chain_data("C:\\Users\\jesme\\OneDrive\\Desktop\\Supply_chain_ml_project\\data\\Processed\\supply_chain_final.csv")

    print("New df shape: ",df_new.shape)
    # print("Feature matrix shape:", X.shape)

if __name__ == "__main__":
    run_pipeline()
