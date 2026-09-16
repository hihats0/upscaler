"""A/V olcum sondasi: eslestirme beklenen gecikme etrafinda yapilmali (2026-09-16).

Flas/bip her 2 sn'de bir, hattin gecikmesi ~1,5 sn: "en yakin" eslestirme -0,5 sn veriyordu."""
from upscaler.watch import AvProbe


def test_pair_uses_expected_offset():
    a = [10.0 + 2.0 * i for i in range(10)]
    b = [t + 1.514 for t in a]
    d = AvProbe.pair(a, b, window=0.5, expected=1.5)
    assert len(d) == 10
    assert all(abs(x - 1.514) < 1e-9 for _, x in d)


def test_pair_default_nearest():
    d = AvProbe.pair([1.0, 3.0], [1.04, 3.05])
    assert [round(x, 3) for _, x in d] == [0.04, 0.05]


def test_slope_split_reports_input_and_output_drift():
    p = AvProbe()
    # Kaynak A/V sabit 40 ms; cikista ses 20 ms/dk kayiyor.
    for i in range(100):
        t = 10.0 + 2.0 * i
        p.flash_in.append(t)
        p.beep_in.append(t + 0.040)
        p.flash_out.append(t + 1.515)
        p.beep_out.append(t + 1.515 + 0.040 + (t - 10.0) / 60 * 0.020)
    s = p.summary(expected_delay=1.5)
    assert abs(s["goruntu_gecikmesi_ms_medyan"] - 1515) < 1
    assert abs(s["kaynak_av_kayma_ms_saatte"]) < 1
    assert abs(s["av_kayma_ms_saatte"] - 1200) < 5
    assert abs(s["ses_gecikmesi_kayma_ms_saatte"] - 1200) < 5
    assert abs(s["goruntu_gecikmesi_kayma_ms_saatte"]) < 1


def test_added_av_ignores_source_steps():
    p = AvProbe()
    for i in range(100):
        t = 10.0 + 2.0 * i
        src_av = 0.063 if i < 50 else 0.042  # ffplay dongu basinda A/V basamagi
        p.flash_in.append(t)
        p.beep_in.append(t + src_av)
        p.flash_out.append(t + 1.510)
        p.beep_out.append(t + src_av + 1.513)
    s = p.summary(expected_delay=1.5)
    lo, med, hi = s["hattin_ekledigi_av_ms_p5_p50_p95"]
    assert abs(med - 3.0) < 0.01 and hi - lo < 0.01
    assert abs(s["hattin_ekledigi_kayma_ms_saatte"]) < 0.5
    assert s["kaynak_av_kayma_ms_saatte"] < -300  # kaynak basamagi egim gibi gorunur
