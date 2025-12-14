import pandas as pd
import glob
import os

# البحث عن كل الملفات CSV
files = glob.glob("*.csv")

print(f"المجلد الحالي: {os.getcwd()}")
print(f"وجدت {len(files)} ملفات: {files}")

all_dfs = []

for file in files:
    city = os.path.basename(file).split('.')[0].capitalize()
    print(f"\nمعالجة {city} من {file}...")

    try:
        # قراءة مع skiprows=8 (كما نجح)
        df = pd.read_csv(file, skiprows=8, encoding='latin1', low_memory=False)
        print(f"الأعمدة المتاحة: {df.columns.tolist()}")

        # تحويل عمود time إلى تاريخ
        df['date'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M', errors='coerce')
        df = df.dropna(subset=['date'])

        # تحويل G(i) و T2m إلى أرقام (مع التعامل مع النصوص)
        df['G(i)'] = pd.to_numeric(df['G(i)'], errors='coerce')  # يحول النصوص إلى NaN
        df['T2m'] = pd.to_numeric(df['T2m'], errors='coerce')

        # إزالة الصفوف التي تحتوي على NaN في الإشعاع أو الحرارة
        df = df.dropna(subset=['G(i)', 'T2m'])

        # تجميع يومي
        daily = df.groupby(df['date'].dt.date).agg({
            'G(i)': 'sum',
            'T2m': 'max'
        }).reset_index()

        # تحويل إلى kWh/m²
        daily['mean_rad'] = daily['G(i)'] / 1000
        daily['temp_max'] = daily['T2m']
        daily['city'] = city
        daily['date'] = pd.to_datetime(daily['date'])

        daily = daily[['date', 'city', 'mean_rad', 'temp_max']]
        all_dfs.append(daily)
        print(f"تم معالجة {city} بنجاح — {len(daily)} يوم")

    except Exception as e:
        print(f"خطأ في {file}: {e}")

# دمج وحفظ
if all_dfs:
    real_df = pd.concat(all_dfs, ignore_index=True)
    real_df = real_df.sort_values(['city', 'date'])
    
    # rolling features
    real_df['rad_rolling'] = real_df.groupby('city')['mean_rad'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    real_df['temp_rolling'] = real_df.groupby('city')['temp_max'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    
    real_df.to_csv("all_cities_real_data.csv", index=False)
    print("\nتم الدمج بنجاح! الملف: all_cities_real_data.csv")
    print(f"المدن: {sorted(real_df['city'].unique())}")
    print(f"عدد الأيام: {len(real_df)}")
    print(real_df.head())
else:
    print("لا توجد بيانات للدمج.")