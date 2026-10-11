import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")


class AllianceCanteenDatasetManager:
    """Creates and manages historical canteen sales data."""

    def __init__(self, filename: str = "alliance_university_canteens.csv"):
        self.filename = filename
        self.canteens = [
            "B.Tech Block Canteen",
            "Law Block Canteen",
            "MBA Block Canteen",
            "Hostel Canteen",
        ]
        self.default_items = [
            {"name": "South Indian Thali", "price": 80, "cost": 40, "peak": "12:30 PM - 02:30 PM"},
            {"name": "North Indian Special Thali", "price": 100, "cost": 50, "peak": "01:00 PM - 03:00 PM"},
            {"name": "Hyderabadi Chicken Biryani", "price": 130, "cost": 70, "peak": "01:00 PM - 02:30 PM"},
            {"name": "Paneer Butter Masala & Roti", "price": 110, "cost": 55, "peak": "01:00 PM - 03:00 PM"},
            {"name": "Samosa & Masala Chai", "price": 30, "cost": 10, "peak": "04:00 PM - 06:00 PM"},
            {"name": "Cold Coffee & Brownie", "price": 90, "cost": 40, "peak": "03:30 PM - 05:30 PM"},
            {"name": "Crispy Masala Dosa", "price": 60, "cost": 25, "peak": "08:00 AM - 10:30 AM"},
            {"name": "Veg Cheese Roll", "price": 70, "cost": 30, "peak": "04:00 PM - 06:30 PM"},
            {"name": "Schezwan Fried Rice", "price": 90, "cost": 45, "peak": "01:00 PM - 03:00 PM"},
            {"name": "Aloo Paratha with Curd", "price": 50, "cost": 20, "peak": "08:30 AM - 11:00 AM"},
            {"name": "Chicken Shawarma", "price": 85, "cost": 40, "peak": "05:00 PM - 08:00 PM"},
            {"name": "Fresh Fruit Juice", "price": 40, "cost": 15, "peak": "11:00 AM - 01:00 PM"},
        ]
        self.df = self._load_or_generate_dataset()

    def _load_or_generate_dataset(self) -> pd.DataFrame:
        if os.path.exists(self.filename):
            try:
                df = pd.read_csv(self.filename)
                required = {
                    "date", "canteen_id", "item_name", "day_of_week",
                    "is_weekend", "is_exam_day", "quantity_sold",
                    "selling_price", "cost_price", "total_revenue",
                    "total_profit", "peak_hour",
                }
                if required.issubset(df.columns) and not df.empty:
                    print(f"Loaded existing dataset: {self.filename}")
                    return df
                print("Existing CSV has an unexpected format. A new dataset will be generated.")
            except Exception as exc:
                print(f"Could not read existing CSV ({exc}). A new dataset will be generated.")

        print("Generating historical sales data for Alliance University canteens...")
        np.random.seed(42)
        records = []
        start_date = datetime(2026, 1, 1)

        for day_offset in range(180):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            is_weekend = int(day_of_week >= 5)
            is_exam_day = int(day_offset % 25 == 0)

            for canteen in self.canteens:
                for item in self.default_items:
                    multiplier = 1.35 if ("B.Tech" in canteen or "Hostel" in canteen) else 0.9
                    demand = int(np.random.randint(30, 110) * multiplier)
                    if is_weekend:
                        demand = int(demand * (1.25 if "Hostel" in canteen else 0.4))
                    if is_exam_day:
                        demand = int(demand * 1.4)

                    qty = max(5, demand)
                    records.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "canteen_id": canteen,
                        "item_name": item["name"],
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "is_exam_day": is_exam_day,
                        "quantity_sold": qty,
                        "selling_price": item["price"],
                        "cost_price": item["cost"],
                        "total_revenue": qty * item["price"],
                        "total_profit": qty * (item["price"] - item["cost"]),
                        "peak_hour": item["peak"],
                    })

        df = pd.DataFrame(records)
        df.to_csv(self.filename, index=False)
        print(f"Generated dataset: {self.filename} ({len(df)} records)")
        return df

    def add_new_item(self, name: str, price: float, cost: float, peak: str) -> None:
        name = name.strip()
        if not name:
            print("Item name cannot be empty.")
            return
        if name in self.df["item_name"].unique():
            print(f"'{name}' already exists in the menu.")
            return
        if price < 0 or cost < 0:
            print("Price and cost cannot be negative.")
            return

        records = []
        start_date = datetime(2026, 1, 1)
        for day_offset in range(180):
            date = start_date + timedelta(days=day_offset)
            day_of_week = date.weekday()
            is_weekend = int(day_of_week >= 5)
            is_exam_day = int(day_offset % 25 == 0)
            for canteen in self.canteens:
                qty = int(np.random.randint(20, 80))
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "canteen_id": canteen,
                    "item_name": name,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_exam_day": is_exam_day,
                    "quantity_sold": qty,
                    "selling_price": price,
                    "cost_price": cost,
                    "total_revenue": qty * price,
                    "total_profit": qty * (price - cost),
                    "peak_hour": peak,
                })

        self.df = pd.concat([self.df, pd.DataFrame(records)], ignore_index=True)
        self.df.to_csv(self.filename, index=False)
        print(f"Successfully added '{name}' to all canteens.")

    def remove_item(self, name: str) -> None:
        if name not in self.df["item_name"].values:
            print(f"Item '{name}' not found.")
            return
        if self.df["item_name"].nunique() <= 1:
            print("Cannot remove the last remaining menu item.")
            return
        self.df = self.df[self.df["item_name"] != name].copy()
        self.df.to_csv(self.filename, index=False)
        print(f"Successfully removed '{name}' from the menu.")

    def get_available_items(self) -> List[str]:
        return sorted(self.df["item_name"].dropna().unique().tolist())

    def get_available_canteens(self) -> List[str]:
        return self.canteens


class AllianceDemandPredictorEngine:
    """Trains the demand prediction model and displays analytics."""

    def __init__(self, dataset_manager: AllianceCanteenDatasetManager):
        self.dm = dataset_manager
        self.retrain_model()

    def retrain_model(self) -> None:
        self.df = self.dm.df.copy()
        if self.df.empty:
            raise ValueError("The dataset is empty. Add or restore food items first.")

        self.total_volume = self.df["quantity_sold"].sum()
        self.total_revenue = self.df["total_revenue"].sum()

        self.df["item_code"] = self.df["item_name"].astype("category").cat.codes
        self.df["canteen_code"] = self.df["canteen_id"].astype("category").cat.codes
        self.item_codes = self.df.groupby("item_name")["item_code"].first().to_dict()
        self.canteen_codes = self.df.groupby("canteen_id")["canteen_code"].first().to_dict()

        features = ["canteen_code", "item_code", "day_of_week", "is_weekend", "is_exam_day"]
        X = self.df[features]
        y = self.df["quantity_sold"]

        if len(self.df) >= 5:
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def analyze_item(self, item_index: int) -> Dict:
        self.retrain_model()
        items = self.dm.get_available_items()
        if item_index < 0 or item_index >= len(items):
            raise IndexError("Invalid food item selection.")

        target_item = items[item_index]
        item_df = self.df[self.df["item_name"] == target_item]
        item_qty = item_df["quantity_sold"].sum()
        item_rev = item_df["total_revenue"].sum()

        qty_share = (item_qty / self.total_volume * 100) if self.total_volume else 0
        rev_share = (item_rev / self.total_revenue * 100) if self.total_revenue else 0
        canteen_breakdown = item_df.groupby("canteen_id")["quantity_sold"].sum().to_dict()

        highest = max(canteen_breakdown, key=canteen_breakdown.get)
        lowest = min(canteen_breakdown, key=canteen_breakdown.get)
        item_code = int(self.item_codes[target_item])
        predictions = {}

        for canteen in self.dm.get_available_canteens():
            canteen_code = self.canteen_codes.get(canteen)
            if canteen_code is None:
                predictions[canteen] = 1
                continue
            row = pd.DataFrame([{
                "canteen_code": canteen_code,
                "item_code": item_code,
                "day_of_week": 1,
                "is_weekend": 0,
                "is_exam_day": 0,
            }])
            predictions[canteen] = max(1, int(round(self.model.predict(row)[0])))

        total_predicted = sum(predictions.values())
        recommended_qty = int(np.ceil(total_predicted * 1.10))
        peak_hour = item_df["peak_hour"].iloc[0]
        price = float(item_df["selling_price"].iloc[0])
        cost = float(item_df["cost_price"].iloc[0])
        profit = recommended_qty * (price - cost)

        return {
            "item_name": target_item,
            "quantity_share": round(qty_share, 2),
            "revenue_share": round(rev_share, 2),
            "canteen_breakdown": canteen_breakdown,
            "highest_demand_canteen": highest,
            "lowest_demand_canteen": lowest,
            "canteen_predictions": predictions,
            "total_predicted_demand": total_predicted,
            "recommended_cook_qty": recommended_qty,
            "peak_hour": peak_hour,
            "unit_price": price,
            "unit_cost": cost,
            "estimated_daily_profit": round(profit, 2),
        }

    def render_analytics_graph(self, report: Dict) -> None:
        plt.close("all")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        canteen_data = pd.Series(report["canteen_breakdown"])
        sns.barplot(x=canteen_data.index, y=canteen_data.values, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Pieces Sold per Canteen: {report['item_name']}")
        axes[0].set_ylabel("Total Pieces Sold")
        axes[0].tick_params(axis="x", rotation=15)
        for patch in axes[0].patches:
            axes[0].annotate(
                f"{int(patch.get_height())}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center", va="bottom", xytext=(0, 4), textcoords="offset points",
            )

        pred_df = pd.DataFrame({
            "Predicted Demand": report["canteen_predictions"],
            "Safety Prep (+10%)": {
                key: int(np.ceil(value * 1.1))
                for key, value in report["canteen_predictions"].items()
            },
        })
        pred_df.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Next-Day ML Cooking Forecast")
        axes[1].set_ylabel("Servings / Pieces")
        axes[1].tick_params(axis="x", rotation=15)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()
        plt.close("all")

    def render_leaderboard_graph(self) -> None:
        plt.close("all")
        summary = self.df.groupby("item_name")["quantity_sold"].sum().sort_values()
        if summary.empty:
            print("No menu data is available to display.")
            return

        plt.figure(figsize=(10, 6))
        median_value = summary.median()
        bar_colors = ["#E74C3C" if value < median_value else "#2ECC71" for value in summary.values]
        plt.barh(summary.index, summary.values, color=bar_colors)
        plt.title("Alliance University Canteen Food Demand Ranking")
        plt.xlabel("Total Pieces Sold Across Campus")
        plt.ylabel("Food Items")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
        plt.close("all")


def main():
    dataset_mgr = AllianceCanteenDatasetManager()
    engine = AllianceDemandPredictorEngine(dataset_mgr)

    while True:
        try:
            available_items = dataset_mgr.get_available_items()

            print("\n" + "=" * 70)
            print("  ALLIANCE UNIVERSITY - SMART CANTEEN DEMAND & GRAPH SYSTEM")
            print("=" * 70)
            print("FOOD ITEMS:")
            for idx, item in enumerate(available_items, 1):
                print(f"[{idx}] {item}")

            print("-" * 70)
            print("[A] Add New Food Item")
            print("[R] Remove Food Item")
            print("[L] View Campus Demand Leaderboard Graph")
            print("[0] Exit Application")
            print("=" * 70)

            choice = input(
                f"Select Option (1-{len(available_items)} or A/R/L/0): "
            ).strip().upper()

            if choice == "0":
                print("\nThank you for using the Smart Canteen System. Goodbye!")
                break

            elif choice == "A":
                print("\nADD NEW FOOD ITEM")
                name = input("Enter Food Item Name: ").strip()
                if not name:
                    print("Item name cannot be empty.")
                    continue
                price = float(input("Enter Selling Price (Rs.): "))
                cost = float(input("Enter Preparation Cost (Rs.): "))
                peak = input("Enter Peak Demand Time: ").strip()

                if price < 0 or cost < 0:
                    print("Price and cost cannot be negative.")
                    continue

                dataset_mgr.add_new_item(name, price, cost, peak)
                engine.retrain_model()

            elif choice == "R":
                print("\nREMOVE FOOD ITEM")
                name = input("Enter Exact Food Item Name to Remove: ").strip()
                dataset_mgr.remove_item(name)
                engine.retrain_model()

            elif choice == "L":
                print("\nDisplaying Campus Demand Leaderboard...")
                engine.render_leaderboard_graph()

            elif choice.isdigit():
                item_index = int(choice) - 1
                if item_index < 0 or item_index >= len(available_items):
                    print("Invalid selection. Choose a food number shown in the menu.")
                    continue

                report = engine.analyze_item(item_index)
                print("\n" + "=" * 70)
                print(f"DEMAND ANALYTICS: {report['item_name'].upper()}")
                print("=" * 70)
                print(f"Share of Campus Food Volume: {report['quantity_share']}%")
                print(f"Share of Campus Revenue: {report['revenue_share']}%")
                print(f"Peak Order Time: {report['peak_hour']}")

                print("\nQUANTITY SOLD BY CANTEEN:")
                for canteen, quantity in report["canteen_breakdown"].items():
                    print(f"  {canteen}: {quantity} pieces")

                print(f"\nHighest Demand Canteen: {report['highest_demand_canteen']}")
                print(f"Lowest Demand Canteen: {report['lowest_demand_canteen']}")

                print("\nNEXT-DAY ML PREDICTED PREPARATION QUANTITIES:")
                for canteen, quantity in report["canteen_predictions"].items():
                    print(f"  {canteen}: Prepare {quantity} servings")

                print("\nCANTEEN OWNER ACTION PLAN")
                print(f"Recommended preparation (including 10% buffer): {report['recommended_cook_qty']} servings")
                print(f"Estimated daily profit: Rs. {report['estimated_daily_profit']:.2f}")
                print(f"Unit cost: Rs. {report['unit_cost']:.2f} | Selling price: Rs. {report['unit_price']:.2f}")

                print("\nRendering visual analytics graphs...")
                engine.render_analytics_graph(report)

            else:
                print("Invalid option. Enter a food number, A, R, L, or 0.")

        except ValueError:
            print("\nInvalid number. Please enter a valid numeric value.")
        except (KeyboardInterrupt, EOFError):
            print("\nInput interrupted. Exiting application.")
            break
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")
            print("Please try another option.")

        print("\nReturning to the main menu...")


if __name__ == "__main__":
    main()
