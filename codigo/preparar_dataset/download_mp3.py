import subprocess


def download_mp3(urls, command):
    for url in urls:
        print(f"\n➡ Descargando: {url}")
        try:
            subprocess.run(command + [url], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error con {url}: código {e.returncode}")

if __name__ == "__main__":
    urls = [
        "https://youtu.be/ZP0SYej47h4?si=IH86nth4XHpQYWwf",
        "https://youtu.be/60dREvamxYw?si=xzlAKnauIDxrLUJP",
        "https://youtu.be/vh82dnK6WxA?si=xDEgEUdln1tR1WdJ"
        "https://youtu.be/jtHpyfUxJPA?si=gVQ_aBKZ7oZGh_of",
        "https://youtu.be/UYsFc5dAT-Y?si=weZl8tt1gPS9Usmw",
        "https://youtu.be/G4SlreDPLJc?si=pwLpzY_5fa55s_7h",
        "https://youtu.be/RK4jV1JV2GA?si=lzKC-OzTX-XSJ5j9",
        "https://youtu.be/zE6JmhuoPI0?si=cwM1bbQ5Jn21uNhZ",
        "https://youtu.be/cCnBmNyHD20?si=TwPrA6RCNksyuAeq",
        "https://youtu.be/Rdo8ungLVzk?si=wtayU7ILIIiSaZBS",
        "https://youtu.be/ZPdbyL1Q-lM?si=84ImEpBSFjoPnvRz",
        "https://youtu.be/YKSl1Lee6_s?si=5T4NYuA81QtqvbUw",
        "https://youtu.be/fdej08BN8nA?si=AjQK7qFdkgyDhfy2",
        "https://youtu.be/1SCauKR3cRE?si=Lc_Mk9PMqv6pWJFB",
        "https://youtu.be/hte4s2x7vqg?si=MoQZlP7_bkK4dD4j",
        "https://youtu.be/2udg-Gb3hJY?si=jP5rBJAKRRFWZoC4",
        "https://youtu.be/xY_th0Z_yGs?si=k423Gkb6UZWNo3hA",
        "https://youtu.be/ptz4P1Zz-aM?si=sSu4skncjGw96uKh",
        "https://youtu.be/XYqOR0ScWmQ?si=TJFVUFgITwp7b0bD",
        "https://youtu.be/ujUzLFFTcuI?si=iMxDHOBNK1Fc6kBX",
        "https://youtu.be/Cg6DYnEXVBU?si=8x932QSQJvM7sdhC",
        "https://youtu.be/BQT6hBgMyXg?si=oQjNerATHuBe8GGU",
        "https://youtu.be/0yRH0m7f0dQ?si=mLGWKqRLzoEUJdLs",
        "https://youtu.be/Q4cXUPAdW30?si=KF6uPT9_Uv8tLNe5",
        "https://youtu.be/e9exMmeRTnI?si=JE7skfBYDPVoE6QX",
        "https://youtu.be/m1NfhdcgBTE?si=Lfh7HWEi4lpgnz-2",
        "https://youtu.be/JfCp72wYAa4?si=PWAuuiG4DA16ECYn",
        "https://youtu.be/9pabuhQr_cA?si=wc5yo7jBlQ0cxvGC",
        "https://youtu.be/njdG63jfff0?si=PhoTHV9oYocV8L0B",
        "https://youtu.be/bxE-t0uDGxM?si=43azE3y45FtZtBxt",
        "https://youtu.be/9mYtpqbgBsc?si=s0U03M6tF1J_t-7h",
        "https://youtu.be/42QUapvb8fg?si=2m0Ulv9qEV7NazdG",
        "https://youtu.be/E9LYmoENx-k?si=TYr2WDYHcAR5YzJL",
        "https://youtu.be/-HKFRVEbe44?si=d8ksyn6BuaGSNIE-",
        "https://youtu.be/cIISx8ymfsE?si=6y-h90n-eATBm1F4",
        "https://youtu.be/pjjeV4UDEPQ?si=VIcTjyzmObMqmvMQ",
        "https://youtu.be/pjjeV4UDEPQ?si=ws835OQYMjQzamkK",
        "https://youtu.be/dJIv9y4i_L0?si=LZ0k9P0hsJhRZiCd",
        "https://youtu.be/q_0QlPGAbcE?si=aVpXv0KwTxefzTvZ",
        "https://youtu.be/yxq6PovcA5k?si=nS4GKKXedIOrIH-y",
        "https://youtu.be/Uh3u_swFDX8?si=aYo7NFKvqo2ulCbZ",
        "https://youtu.be/7puRTLyAxiU?si=_Vja7wJrOUGduN8O",
        "https://youtu.be/rl9VHMV1Wx8?si=MvCYwZ2Z-zcJAZA7",
        "https://youtu.be/EgHMYR4nEWU?si=jXIhwmIM8Z6IZLVb",
        "https://youtu.be/IypehETlBKU?si=HOW5RAZaUd3oC0-_",
        "https://youtu.be/jg080y-WbWA?si=mQaHq5vp0UJyGYeH",
        "https://youtu.be/mnkaugSwfqI?si=SELkZubAZdrqiVfy",
        "https://youtu.be/xp41qJ3MzOI?si=y5D3p0BECTo8chOB",
        "https://youtu.be/nSwXHve_NFQ?si=wWyloNtkp_65CssN",


        "https://youtu.be/nxUAsQlMn2Q?si=AtgyqPkUYxMRjY-X",
        "https://youtu.be/NRgZ6RCmoxQ?si=ZSxOLn8SsJsUdsF_",
        "https://youtu.be/uJUMVWz8TdA?si=lnm-Edc-SvO5oWoR",
        "https://youtu.be/jiBA9nwVfWU?si=ebp85WSTYoDEwzsy",
        "https://youtu.be/jxkj1xprGys?si=8qH0IFlPUaHyuWJG",
        "https://youtu.be/QXxxl8ckqDY?si=1bpSGqyRh69OpfUH",
        "https://youtu.be/GtOaDDac0Tk?si=3Y236xS_8pmR8ZCr",
        "https://youtu.be/lf0EODTOhV0?si=LWv4OsZ8ty35OlVf",
        "https://youtu.be/h1JjHoy-xHI?si=jgwzIR6_bPOLsSpl",
        "https://youtu.be/cvofPcZn1J0?si=T-iBNZAxHy_VtVb6",
        "https://youtu.be/tQTprgZ9-7A?si=xtPKz62Ox-0CoJLS",
        "https://youtu.be/a3IUksuPVA4?si=xjuLevjHsa9bPuW6",
        "https://youtu.be/y1a_I3TcoDk?si=pW46UH0OQUOyCkvu",
        "https://youtu.be/OVbNGRB6EPg?si=U7JZmBXsdiQ_427o",
        "https://youtu.be/zUSmU0EXRgU?si=zlOtsBM0pJhGkyBt",

    ]
    command = ["yt-dlp", "-x", "--audio-format", "mp3"]
    download_mp3(urls, command)
