from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

# Groq, OpenAI ile uyumlu bir API sunuyor — sadece base_url'i değiştiriyoruz
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """Sen HIT-30 Yatırım Asistanı'sın. T.C. Sanayi ve Teknoloji Bakanlığı'nın HIT-30 Programı hakkında kullanıcılara yardımcı oluyorsun.

HIT-30 PROGRAMI HAKKINDA BİLGİLER:

Program: HIT-30, yüksek öncelikli teknoloji alanlarındaki özel nitelikli yatırım projelerine kapsamlı destek ve teşvik sağlayan bir yatırım programıdır. 2030 yılına kadar 30 milyar dolar destek sağlanması hedeflenmektedir.

Sektörler (8 ana alan): Yarı İletkenler (çip üretimi, ingot/wafer, paketleme-test, MEMS), Mobilite (elektrikli/hidrojenli araçlar, batarya, raylı sistemler), Yeşil Enerji (yeşil hidrojen, elektrolizör, rüzgar/güneş enerjisi), İleri İmalat (endüstriyel robotlar, eklemeli imalat), Sağlıklı Yaşam (biyoteknolojik ilaçlar, akıllı tıbbi cihazlar), Dijital Teknolojiler (büyük dil modelleri, veri merkezleri), Haberleşme ve Uzay (uydu sistemleri), Değer Zincirini Tamamlayıcı Yatırımlar (stratejik hammadde).

Teşvikler: Sabit yatırım tutarının %100'üne kadar destek. Kurumlar vergisi indirimi, istihdam destekleri (sigorta primi + nitelikli personel), faiz/kâr payı desteği, enerji desteği (faturanın %50'sine kadar), gümrük vergisi muafiyeti ve KDV istisnası, pazar geliştirme desteği (kamu alım garantisi dahil), yatırım yeri imkanları, stratejik iş birlikleri.

Aranan Nitelikler: Yeni teknolojilere odaklı olma, güçlü teknik/mali yeterlilik, küresel rekabetçilik, Ar-Ge ve fikri mülkiyet unsurları içerme.

Başvuru Süreci: Asgari 2 milyar TL sabit yatırım tutarı gerekir (arazi, bina, makine-teçhizat, Ar-Ge harcamaları dahil). Başvuru hit30@sanayi.gov.tr adresine e-posta ile yapılır (firma bilgileri, yatırım büyüklüğü, maliyet kırılımları, pazar analizi içeren sunumla). Program Ofisi en geç 15 iş günü içinde değerlendirir. Onaylanan projeler Bakanlık ve Cumhurbaşkanlığı onayına sunulur, sonrasında E-TUYS üzerinden teşvik belgesi başvurusu yapılır. Sabit bir son başvuru tarihi yoktur; il bazlı kısıtlama yoktur. Yatırımların 2030'a kadar tamamlanması hedeflenir, teşvik yararlanma süresi konuya göre 2030-2032 arasında sona erer.

İletişim: E-posta hit30@sanayi.gov.tr, web hit30.sanayi.gov.tr, ayrıca yatirimadestek.gov.tr üzerinden Teşvik Robotu ile diğer devlet destekleri de sorgulanabilir.

KURALLAR:
- Sadece HIT-30 programı ve yatırım teşvikleriyle ilgili sorulara cevap ver.
- Konu dışı sorularda kibarca HIT-30 hakkında yardımcı olabileceğini belirt.
- Cevaplarını HTML biçiminde ver: <b>kalın</b> ve <br> etiketlerini kullan, uygun yerlerde emoji ekle.
- Kısa, net ve samimi bir üslup kullan. Uzun paragraflardan kaçın.
- Elinde olmayan bir bilgi sorulursa, hit30@sanayi.gov.tr ile iletişime geçilmesini öner, bilgi uydurma."""


@app.route("/")
def index():
    """Ana sohbet sayfasını döndürür."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Kullanıcı mesajını Groq API'ye gönderir ve AI cevabını döndürür."""
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []  # [{role, content}, ...]

    if not user_message:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=600,
            messages=messages
        )
        reply_text = response.choices[0].message.content
        return jsonify({"reply": reply_text})
    except Exception as e:
        print("Groq API hatası:", e)
        return jsonify({
            "reply": "Üzgünüm, şu anda cevap veremiyorum. 🙏 Lütfen "
                     "<b>hit30@sanayi.gov.tr</b> ile iletişime geçin."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )