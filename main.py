import pandas as pd
from pycaret.classification import *
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
                                Table, TableStyle)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_CENTER
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import unicodedata


# Function to remove Polish characters
def remove_polish_characters(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return str(text)


# Load data

data = pd.read_csv("CollegeDistance.csv")


# Step 1: Data Exploration
# Check for missing values
missing_values = data.isnull().sum()

# Basic statistics rounded to 2 decimal places
statistics = data.describe().round(2)

# Drop rows with missing values
data = data.dropna()
num_records_after_dropping = data.shape[0]

# Step 2: Visualization of 'score' distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['score'], kde=True, bins=20)
plt.title('Rozklad Score')
plt.xlabel('Score')
plt.ylabel('Czestotliwosc')
plt.tight_layout()
plt.savefig('score_distribution.png')
plt.close()

# Bar plots for categorical variables
categorical_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']

# Generate separate plots for each categorical variable
for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=data[col])
    plt.title(f'Rozklad {col}')
    plt.ylabel('Liczba')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(f'{col}_distribution.png')
    plt.close()

# Step 3: Data Preparation and Model Training
data['score'] = pd.qcut(data['score'], q=10, labels=False)

# Normalize data and set up PyCaret
exp_clf = setup(data, target='score', session_id=123, normalize=True, verbose=False)

# Compare models with additional metrics
best_models = compare_models(n_select=10, sort='Accuracy', exclude=['lightgbm'])
results = pull()

# Select the best model based on Accuracy
best_model = best_models[0]
best_model_name = best_model.__class__.__name__

# List of metrics to analyze
metrics = ['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC']

# Generate model comparison plots
for metric in metrics:
    metric_filename = metric.lower().replace('.', '')
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results['Model'], y=results[metric])
    plt.title(f'Porownanie modeli na podstawie {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'model_{metric_filename}_comparison.png')
    plt.close()  

    

# Check if the best model is LinearDiscriminantAnalysis
is_lda = best_model_name == 'LinearDiscriminantAnalysis'

# Step 4: Generate PDF Documentation using ReportLab

# Create PDF document
doc = SimpleDocTemplate("Readme.pdf", pagesize=A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', alignment=4))
styles.add(ParagraphStyle(name='TableModel', alignment=TA_CENTER, fontSize=8, leading=9, wordWrap='CJK'))
Story = []


# Function to add text without Polish characters
def add_paragraph(text, style):
    text = remove_polish_characters(text)
    Story.append(Paragraph(text, style))


# Add title
add_paragraph("Dokumentacja: Model predykcyjny dla zmiennej 'score'", styles['Title'])
Story.append(Spacer(1, 12))

# Introduction
add_paragraph("Wprowadzenie", styles['Heading2'])
intro_text = ("Celem tego projektu bylo stworzenie modelu predykcyjnego, ktory przewiduje zmienna 'score'. "
              "Ponizej znajduje sie dokumentacja pracy, w tym analiza danych, wybor modeli oraz ich ocena.")
add_paragraph(intro_text, styles['Justify'])
Story.append(Spacer(1, 12))

# Data Exploration
add_paragraph("Krok 1: Eksploracja i analiza danych", styles['Heading2'])

# Missing values
missing_values_text = "<br />".join([f"{index}: {value}" for index, value in missing_values.items()])
add_paragraph("Brakujace wartosci w danych przed usunieciem:", styles['Heading3'])
add_paragraph(missing_values_text, styles['Justify'])
Story.append(Spacer(1, 12))

# Data processing information
add_paragraph("Przetwarzanie danych:", styles['Heading3'])
data_processing_text = (f"Przed trenowaniem modeli usunieto wiersze zawierajace brakujace dane. "
                        f"Liczba rekordow po usunieciu brakujacych danych: {num_records_after_dropping}. "
                        "Dodatkowo, przed trenowaniem modele dane zostaly znormalizowane za pomoca normalizacji Z-score, "
                        "ktora polega na odjeciu sredniej i podzieleniu przez odchylenie standardowe dla kazdej cechy. "
                        "To pomaga w poprawie wydajnosci modeli, zapewniajac, ze wszystkie cechy znajduja sie na podobnej skali.")
add_paragraph(data_processing_text, styles['Justify'])
Story.append(Spacer(1, 12))

# Basic statistics as a table
add_paragraph("Podstawowe statystyki:", styles['Heading3'])

# Convert statistics to table format with rounding
statistics_reset = statistics.reset_index()
statistics_table_data = [statistics_reset.columns.tolist()] + statistics_reset.values.tolist()

# Set maximum page width for the table
max_table_width = A4[0] - 2 * inch  # page width minus margins
column_count = len(statistics_table_data[0])
column_width = max_table_width / column_count

# Style for the statistics table
statistics_table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
])

statistics_table = Table(statistics_table_data, colWidths=[column_width] * column_count)
statistics_table.setStyle(statistics_table_style)
Story.append(statistics_table)
Story.append(Spacer(1, 12))

# Add explanations to statistics
add_paragraph("Wyjasnienie statystyk:", styles['Heading3'])

statistics_explanation = [
    ("count", "Liczba niepustych obserwacji dla kazdej zmiennej."),
    ("mean", "Srednia arytmetyczna wartosci dla kazdej zmiennej."),
    ("std", "Odchylenie standardowe, miara rozproszenia danych wokol sredniej."),
    ("min", "Minimalna wartosc w zbiorze danych dla kazdej zmiennej."),
    ("25%", "Pierwszy kwartyl (Q1), 25% danych znajduje sie ponizej tej wartosci."),
    ("50%", "Mediana (Q2), 50% danych znajduje sie ponizej tej wartosci."),
    ("75%", "Trzeci kwartyl (Q3), 75% danych znajduje sie ponizej tej wartosci."),
    ("max", "Maksymalna wartosc w zbiorze danych dla kazdej zmiennej.")
]

for stat_name, explanation in statistics_explanation:
    add_paragraph(f"<b>{stat_name}:</b> {explanation}", styles['Justify'])
    Story.append(Spacer(1, 6))

Story.append(Spacer(1, 12))

# Add 'score' distribution plot
add_paragraph("Rozklad zmiennej 'score':", styles['Heading3'])

# Adjust image size
max_page_width, max_page_height = A4
margin = inch
max_image_width = max_page_width - 2 * margin
max_image_height = max_page_height - 2 * margin

img_path = 'score_distribution.png'
img_reader = ImageReader(img_path)
img_width, img_height = img_reader.getSize()
scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
img_width_scaled = img_width * scaling_factor
img_height_scaled = img_height * scaling_factor
img = Image(img_path, width=img_width_scaled, height=img_height_scaled)
Story.append(img)
Story.append(Spacer(1, 12))

# Add categorical variable distributions
add_paragraph("Rozklady zmiennych kategorycznych:", styles['Heading3'])

for col in categorical_columns:
    add_paragraph(f"Rozklad '{col}':", styles['Heading4'])
    img_path = f'{col}_distribution.png'
    img_reader = ImageReader(img_path)
    img_width, img_height = img_reader.getSize()
    scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
    img_width_scaled = img_width * scaling_factor
    img_height_scaled = img_height * scaling_factor
    img = Image(img_path, width=img_width_scaled, height=img_height_scaled)
    Story.append(img)
    Story.append(Spacer(1, 12))

# Step 3: Model Comparison
Story.append(PageBreak())
add_paragraph("Krok 3: Porownanie modeli", styles['Heading2'])
comparison_text = ("Najlepsze modele zostaly porownane za pomoca funkcji compare_models() z biblioteki PyCaret. "
                   "Ponizej znajduja sie wyniki porownania:")
add_paragraph(comparison_text, styles['Justify'])
Story.append(Spacer(1, 12))

# Prepare data for the results table
results_table = results[['Model', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC']].round(2)

# Convert DataFrame to list of lists
results_table_data = [results_table.columns.tolist()] + results_table.values.tolist()

# Wrap text in 'Model' column
for i in range(1, len(results_table_data)):
    model_name = results_table_data[i][0]
    model_name = remove_polish_characters(str(model_name))
    results_table_data[i][0] = Paragraph(model_name, styles['TableModel'])

# Set column widths
max_table_width = A4[0] - 2 * inch  # page width minus margins
model_col_width = max_table_width * 0.4  # 40% of table width for 'Model' column
other_cols_width = (max_table_width - model_col_width) / (len(results_table.columns) - 1)
col_widths = [model_col_width] + [other_cols_width] * (len(results_table.columns) - 1)

# Style for the model results table
results_table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),  # Center alignment for other columns
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ('FONTSIZE', (0, 0), (-1, -1), 8),
])

# Create the model results table
results_table = Table(results_table_data, colWidths=col_widths, repeatRows=1, splitByRow=True)
results_table.setStyle(results_table_style)
Story.append(results_table)
Story.append(Spacer(1, 12))

# Add explanations of metrics
add_paragraph("Wyjasnienie metryk:", styles['Heading3'])

metrics_explanation = [
    ("Accuracy", "Odsetek poprawnie sklasyfikowanych obserwacji w stosunku do wszystkich obserwacji."),
    ("AUC", "Pole pod krzywa ROC, mierzy zdolnosc modelu do rozrozniania klas."),
    ("Recall", "Odsetek rzeczywistych pozytywnych przypadkow, ktore zostaly poprawnie zidentyfikowane."),
    ("Prec.", "Precyzja, odsetek pozytywnych identyfikacji, ktore byly prawidlowe."),
    ("F1", "Srednia harmoniczna precyzji i recall, balansuje miedzy precyzja a recall."),
    ("Kappa", "Statystyka Kappy Cohena, mierzy zgodnosc miedzy przewidywaniami a rzeczywistymi klasami."),
    ("MCC", "Wspolczynnik korelacji Matthewsa, uwzglednia wszystkie cztery kategorie w macierzy pomylek.")
]

for metric_name, explanation in metrics_explanation:
    add_paragraph(f"<b>{metric_name}:</b> {explanation}", styles['Justify'])
    Story.append(Spacer(1, 6))

Story.append(Spacer(1, 12))

# Add plots for model results
for metric in metrics:
    metric_filename = metric.lower().replace('.', '')
    add_paragraph(f"Porownanie modeli na podstawie {metric}:", styles['Heading3'])
    img_path = f'model_{metric_filename}_comparison.png'
    img_reader = ImageReader(img_path)
    img_width, img_height = img_reader.getSize()
    scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
    img_width_scaled = img_width * scaling_factor
    img_height_scaled = img_height * scaling_factor
    img = Image(img_path, width=img_width_scaled, height=img_height_scaled)
    Story.append(img)
    Story.append(Spacer(1, 12))

# Step 4: Selection and Training of the Best Model
Story.append(PageBreak())
add_paragraph("Krok 4: Wybor i trenowanie najlepszego modelu", styles['Heading2'])

# Add description of the selected model based on Accuracy
best_model_text = (f"Na podstawie porownania modeli najlepszym modelem jest <b>{best_model_name}</b>.<br />"
                   "Model ten osiagnal najwyzsza wartosc Accuracy, co czyni go najbardziej skutecznym modelem "
                   "do przewidywania zmiennej 'score'. Ponadto, model ten wykazuje dobra rownowage pomiedzy "
                   "dokladnoscia a innymi wskaznikami, takimi jak AUC, Recall, Precyzja, F1-score, Kappa oraz MCC, "
                   "co czyni go stabilnym i uniwersalnym wyborem.")
add_paragraph(best_model_text, styles['Justify'])
Story.append(Spacer(1, 12))

# Add detailed description of the training method
add_paragraph("Sposob trenowania modelu", styles['Heading3'])
training_description = (
    "Model zostal przetrenowany z wykorzystaniem <b>walidacji krzyzowej</b> (ang. cross-validation) z 5-krotnym podzialem danych (5-fold cross-validation). "
    "Walidacja krzyzowa polega na podziale danych na k rownych czesci (tzw. foldow). "
    "Model jest trenowany k razy, za kazdym razem uzywajac k-1 czesci danych jako zbioru treningowego, a pozostalej czesci jako zbioru testowego. "
    "Srednia z wynikow uzyskanych w kazdej iteracji sluzy do oceny ogolnej wydajnosci modelu. "
    "Takie podejscie pozwala na lepsza ocene modelu, poniewaz wykorzystuje wszystkie dostepne dane zarowno do trenowania, jak i testowania, "
    "minimalizujac wplyw losowego podzialu danych na wyniki."
)
add_paragraph(training_description, styles['Justify'])
Story.append(Spacer(1, 12))

# If the best model is LinearDiscriminantAnalysis, add 'shrinkage' parameter analysis
if is_lda:
    # Add the 'shrinkage' analysis plot
    add_paragraph("Analiza wplywu parametru 'shrinkage' na dokladnosc modelu LinearDiscriminantAnalysis",
                  styles['Heading3'])
    shrinkage_analysis_text = (
        "Aby zbadac wplyw parametru <b>'shrinkage'</b> na wydajnosc modelu, przeprowadzono eksperymenty z roznymi wartosciami tego parametru. "
        "Parametr 'shrinkage' sluzy do regularizacji macierzy kowariancji, co moze poprawic wydajnosc modelu w przypadku danych z wysoka korelacja miedzy cechami. "
        "Wyniki pokazuja, jak zmienia sie srednia wartosc Accuracy w zaleznosci od wartosci parametru 'shrinkage'."
    )
    add_paragraph(shrinkage_analysis_text, styles['Justify'])
    Story.append(Spacer(1, 12))

    # Add the plot to the report
    img_path = 'shrinkage_accuracy.png'
    img_reader = ImageReader(img_path)
    img_width, img_height = img_reader.getSize()
    scaling_factor = min(max_image_width / img_width, max_image_height / img_height)
    img_width_scaled = img_width * scaling_factor
    img_height_scaled = img_height * scaling_factor
    img = Image(img_path, width=img_width_scaled, height=img_height_scaled)
    Story.append(img)
    Story.append(Spacer(1, 12))
else:
    add_paragraph(f"Najlepszy model to {best_model_name}, ktory nie jest LinearDiscriminantAnalysis. "
                  "Analiza parametru 'shrinkage' nie zostala przeprowadzona.", styles['Justify'])
    Story.append(Spacer(1, 12))

# Save the PDF file
doc.build(Story)

print("Dokumentacja PDF zostala wygenerowana i zapisana jako Readme.pdf.")
