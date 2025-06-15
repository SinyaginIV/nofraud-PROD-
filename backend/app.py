from flask import Flask, request, jsonify, send_file, send_from_directory
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # отключает GUI-бэкенд
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate, Image, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
pdfmetrics.registerFont(TTFont('DejaVu', 'static/fonts/DejaVuSans.ttf'))

app = Flask(__name__, static_url_path='/static', static_folder='static')
PROCESSED_DIR = 'processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ======= ГЕНЕРАЦИЯ ГРАФИКА nofraud!Snap =======
def generate_nofraud_snap(ids, y_pred_proba, fraud_flags=None, X=None, model=None, features=None, threshold=0.5):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(14, 7))
    ids = np.array(ids)
    y_pred_proba = np.array(y_pred_proba)
    if fraud_flags is None:
        fraud_flags = np.zeros_like(y_pred_proba)

    # SHAP stacked bar
    try:
        if X is not None and model is not None:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            feature_names = list(X.columns if features is None else features)
            if isinstance(shap_values, list):  # RandomForest: shap_values[1] — для класса Fraud
                shap_values = shap_values[1]
            elif shap_values.ndim == 3:
                # (n_samples, n_features, n_classes)
                shap_values = shap_values[:, :, 1]
            print("ids shape:", len(ids))
            print("shap_values shape:", shap_values.shape)
            bottom = np.zeros(len(X))
            colors = plt.get_cmap('tab10').colors
            for i, feature in enumerate(feature_names):
                plt.bar(ids, shap_values[:, i], bottom=bottom, color=colors[i % 10], label=feature if i < 10 else None, alpha=0.7)
                bottom += shap_values[:, i]
            plt.legend()
    except Exception as ex:
        print("SHAP визуализация не удалась:", ex)

    colors = ['#21b563' if not is_fraud else '#ea3a24' for is_fraud in fraud_flags]
    plt.bar(ids, y_pred_proba, color=colors, alpha=0.25, label='Score (столбцы)')
    plt.plot(ids, y_pred_proba, 'o-', color='royalblue', label="Вероятность (score)")
    plt.axhline(threshold, color='red', linestyle='--', label='Порог')
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    for x, y, f in zip(ids, y_pred_proba, fraud_flags):
        if f:
            plt.text(x, y + 0.04, 'Fraud', ha='center', color='#ea3a24', fontsize=9, fontweight='bold')
    plt.xlabel('Порядок транзакций')
    plt.ylabel('Вероятность мошшенической транзакции SHAP + SCORE')
    plt.title('nofraud!Snap')
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig('static/nofraud_snap.png')
    plt.close()

# ======= ГЕНЕРАЦИЯ PDF-ОТЧЁТА =======
def generate_report_pdf(
        output_pdf, 
        df, 
        logo_path='static/logo_nofraud.png', 
        snap_path='static/nofraud_snap.png', 
        input_filename="Файл не указан"
    ):
    from datetime import datetime
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    style_center = ParagraphStyle('centered', parent=styles['Title'], alignment=TA_CENTER, fontName='DejaVu')
    style_normal_center = ParagraphStyle('centeredNormal', parent=styles['Normal'], alignment=TA_CENTER, fontName='DejaVu')
    style_normal = ParagraphStyle('customNormal', parent=styles['Normal'], fontName='DejaVu')
    style_h2 = ParagraphStyle('CustomHeading2', parent=styles['Heading2'], fontName='DejaVu')

    # --- Первая страница ---
    if os.path.exists(logo_path):
        img = Image(logo_path)
        img.drawHeight = 180
        img.drawWidth = 180
        img.hAlign = 'CENTER'
        story.append(Spacer(1, 110))
        story.append(img)
    else:
        story.append(Spacer(1, 200))
    story.append(Spacer(1, 60))
    story.append(Paragraph('<b>nofraud! by Sinyagin Ilya</b>', style_center))
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        f"<b>Дата создания отчёта:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_normal_center))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Проверяемый файл:</b> {input_filename}", style_normal_center))
    story.append(Spacer(1, 70))
    story.append(Paragraph(" ", style_normal_center))
    story.append(PageBreak())

    # --- Вторая страница: только Fraud == 1 ---
    frauds = df[df['Fraud'] == 1]
    if frauds.empty:
        story.append(Paragraph("Мошеннических операций не обнаружено.", style_center))
    else:
        story.append(Paragraph(
            f"<b>Обнаружено мошеннических транзакций: {len(frauds)} из {len(df)}</b>", style_h2))
        story.append(Spacer(1, 12))
        fraud_cols = list(frauds.columns)
        for i, row in frauds.iterrows():
            story.append(Spacer(1, 10))
            block = []
            for col in fraud_cols:
                if col == 'Fraud':
                    continue
                val = row[col]
                if col.lower() == 'reason' and val:
                    block.append(Paragraph(f'<b>Причина:</b> <span color="red">{val}</span>', style_normal))
                else:
                    block.append(Paragraph(f"<b>{col}:</b> {val}", style_normal))
            fraud_table = Table([[b] for b in block], style=[
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVu'),
                ('BOX', (0,0), (-1,-1), 1, colors.red),
                ('INNERGRID', (0,0), (-1,-1), 0.25, colors.red),
                ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ], hAlign='LEFT', colWidths=[440])
            story.append(fraud_table)
            story.append(Spacer(1, 6))

    if os.path.exists(snap_path):
        story.append(PageBreak())
        story.append(Paragraph("Визуализация вероятности мошенничества:", style_center))
        story.append(Image(snap_path, width=440, height=220))
        story.append(Spacer(1, 12))

    doc.build(story)
    doc.title = "nofraud!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/download', methods=['GET'])
def download():
    files = sorted(Path(PROCESSED_DIR).glob('result_*.csv'), key=os.path.getmtime, reverse=True)
    return send_file(files[0], as_attachment=True)

@app.route('/report', methods=['GET'])
def report():
    files = sorted(Path(PROCESSED_DIR).glob('report_*.pdf'), key=os.path.getmtime, reverse=True)
    return send_file(files[0], as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        model_type = request.form.get('modelType', 'test').lower()
        
        if model_type == 'new':
            model_path = 'model/new_model.pkl'
            features_path = 'model/new_model_features.pkl'
            features = joblib.load(features_path)
            if 'Fraud' in df.columns:
                df = df.drop(columns=['Fraud'])

            needed_cols = set(features + ['DateTime'])
            for col in features:
                if col not in df.columns:
                    return jsonify({'error': f'Нет столбца {col}.'}), 400
            extra_cols = set(df.columns) - needed_cols
            if extra_cols:
                return jsonify({'error': f'Лишние столбцы: {", ".join(extra_cols)}'}), 400

            model = joblib.load(model_path)
            score = model.predict_proba(df[features])[:, 1]
            threshold = 0.5
            df['Fraud'] = (score > threshold).astype(int)
            output_cols = ['DateTime'] + features + ['Fraud']
            df = df[output_cols]
            X = df[features]
            ids = range(1, len(df)+1)
            generate_nofraud_snap(
                ids,
                score,
                fraud_flags=df['Fraud'].values,
                X=X,
                model=model,
                features=features
            )
            
        else:
            required_cols = [
                'DateTime', 'Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight'
            ]
            for col in required_cols:
                if col not in df.columns:
                    return jsonify({'error': f'Нет столбца {col}.'}), 400
            extra_cols = set(df.columns) - set(required_cols)
            if extra_cols:
                return jsonify({'error': f'Лишние столбцы: {", ".join(extra_cols)}'}), 400

            le_region = joblib.load('model/le_region.pkl')
            le_device = joblib.load('model/le_device.pkl')
            df['Region'] = le_region.transform(df['Region'])
            df['DeviceType'] = le_device.transform(df['DeviceType'])

            features = [
                'Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight'
            ]
            model = joblib.load('model/real_model.pkl')
            score = model.predict_proba(df[features])[:, 1]
            threshold = 0.5
            df['Fraud'] = (score > threshold).astype(int)
            def explain(row):
                reasons = []
                if row['Amount'] > 4000:
                    reasons.append('Сумма > 4000')
                if row['IsNight'] == 1:
                    reasons.append('Ночная транзакция')
                if row['IsAbroad'] == 1:
                    reasons.append('Заграничная транзакция')
                if row['TxCountLastHour'] > 5:
                    reasons.append('Много транзакций в течение часа')
                return '; '.join(reasons) if row['Fraud'] == 1 else ''
            df['Reason'] = df.apply(explain, axis=1)
            output_cols = [
                'DateTime', 'Amount', 'Region', 'DeviceType', 'IsAbroad', 'TxCountLastHour', 'IsNight', 'Fraud', 'Reason'
            ]
            df = df[output_cols]
            generate_nofraud_snap(
                range(1, len(df)+1),
                score,
                fraud_flags=df['Fraud'].values
            )

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        output_csv = f'{PROCESSED_DIR}/result_{timestamp}.csv'
        output_pdf = f'{PROCESSED_DIR}/report_{timestamp}.pdf'
        df.to_csv(output_csv, index=False, encoding='utf-8-sig', sep=';')
        input_filename = getattr(file, 'filename', 'Файл не указан')
        generate_report_pdf(output_pdf, df, input_filename=input_filename)

        return jsonify({
            'columns': output_cols,
            'rows': df.values.tolist(),
            'fraud_count': int(df["Fraud"].sum())
        })
    except Exception as e:
        print("ОШИБКА:", e)
        return jsonify({'error': str(e)})

@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        features = [col for col in df.columns if col not in ['DateTime', 'Fraud']]
        if 'Fraud' not in df.columns or len(features) < 5:
            return jsonify({'error': 'Для обучения нужны минимум 5 признаков + DateTime + Fraud.'}), 400
        
        joblib.dump(features, 'model/new_model_features.pkl')
        X = df[features]
        y = df['Fraud']
        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X, y)
        joblib.dump(model_new, 'model/new_model.pkl')
        return jsonify({'message': 'Новая модель успешно обучена.'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обучения: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)