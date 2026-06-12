/*
 * Paralel Convex Polygon Kontrol Sistemi
 * Grup 2 - OpenMP ile Cross Product Paralellestirme
 *
 * Derleme (MSVC): cl.exe /openmp /D_USE_MATH_DEFINES /Zi /EHsc /nologo /Fegrp2.exe grp2.c
 * Derleme (GCC) : gcc -fopenmp -O2 -o grp2 grp2.c -lm
 */

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double x;
    double y;
} Nokta;

/*
 * Cross Product
 * Pozitif → CCW (sola donus)
 * Negatif → CW  (saga donus)
 */
double cross(Nokta a, Nokta b, Nokta c) {
    return (b.x - a.x) * (c.y - b.y) -
           (b.y - a.y) * (c.x - b.x);
}

/* ─────────────────────────────────────────────
 * SIRALI versiyon (olcum referansi)
 * ───────────────────────────────────────────── */
int convex_sirali(Nokta *p, int n) {
    int pozitif = 0, negatif = 0;
    int i;
    for (i = 0; i < n; i++) {
        double cp = cross(p[i], p[(i+1)%n], p[(i+2)%n]);
        if (cp > 0) pozitif = 1;
        if (cp < 0) negatif = 1;
        if (pozitif && negatif) return 0;
    }
    return 1;
}

/* ─────────────────────────────────────────────
 * PARALEL versiyon - MSVC OpenMP 2.0 uyumlu
 *
 * Race condition cozumu:
 *   omp_lock_t kullanilir.
 *   Her thread cross product hesaplar (bagimsiz).
 *   Concave tespit edilince lock alip paylasilan
 *   degiskene yazar, lock birakir.
 *   Boylece iki thread ayni anda yazamaz.
 * ───────────────────────────────────────────── */
int convex_paralel(Nokta *p, int n, int thread_sayisi) {
    int concave_bulundu = 0;
    int i;
    double cp_ilk = cross(p[0], p[1], p[2]);
    omp_lock_t kilit;
    omp_init_lock(&kilit);

    omp_set_num_threads(thread_sayisi);

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        double cp = cross(p[i], p[(i+1)%n], p[(i+2)%n]);

        if ((cp > 0 && cp_ilk < 0) || (cp < 0 && cp_ilk > 0)) {
            omp_set_lock(&kilit);
            concave_bulundu = 1;
            omp_unset_lock(&kilit);
        }
    }

    omp_destroy_lock(&kilit);
    return !concave_bulundu;
}

/* ─────────────────────────────────────────────
 * Duzgun n-gen polygon olustur (her zaman convex)
 * ───────────────────────────────────────────── */
void duggun_polygon_olustur(Nokta *p, int n) {
    int i;
    for (i = 0; i < n; i++) {
        p[i].x = cos(2.0 * M_PI * i / n) * 100.0;
        p[i].y = sin(2.0 * M_PI * i / n) * 100.0;
    }
}

/* ─────────────────────────────────────────────
 * Hizlanma tablosu
 * ───────────────────────────────────────────── */
void hizlanma_tablosu(Nokta *p, int n, int tekrar) {
    double sure_sirali = 0.0, sure_paralel = 0.0;
    double t0, t1;
    int r, t;

    for (r = 0; r < tekrar; r++) {
        t0 = omp_get_wtime();
        convex_sirali(p, n);
        t1 = omp_get_wtime();
        sure_sirali += (t1 - t0);
    }
    sure_sirali /= tekrar;

    printf("\n--- %6d nokta | Sirali: %.7f sn ---\n", n, sure_sirali);
    printf("  Thread |   Paralel sure  | Hizlanma\n");
    printf("  -------|-----------------|----------\n");

    {
        int thread_dizi[4];
        thread_dizi[0] = 2;
        thread_dizi[1] = 4;
        thread_dizi[2] = 8;
        thread_dizi[3] = 16;

        for (t = 0; t < 4; t++) {
            double hizlanma;
            sure_paralel = 0.0;
            for (r = 0; r < tekrar; r++) {
                t0 = omp_get_wtime();
                convex_paralel(p, n, thread_dizi[t]);
                t1 = omp_get_wtime();
                sure_paralel += (t1 - t0);
            }
            sure_paralel /= tekrar;
            hizlanma = (sure_paralel > 0.0) ? (sure_sirali / sure_paralel) : 0.0;
            printf("  %6d | %13.7f sn | %.2fx\n",
                   thread_dizi[t], sure_paralel, hizlanma);
        }
    }
}

/* ─────────────────────────────────────────────
 * Dogruluk testleri
 * ───────────────────────────────────────────── */
void manuel_test() {
    int n = 5;
    Nokta cp[5];
    Nokta cc[5];

    cp[0].x=0; cp[0].y=0;
    cp[1].x=4; cp[1].y=0;
    cp[2].x=5; cp[2].y=3;
    cp[3].x=2; cp[3].y=5;
    cp[4].x=0; cp[4].y=3;

    cc[0].x=0; cc[0].y=0;
    cc[1].x=4; cc[1].y=0;
    cc[2].x=2; cc[2].y=1;
    cc[3].x=4; cc[3].y=4;
    cc[4].x=0; cc[4].y=4;

    printf("=== Test 1: Convex (beklenen: Convex) ===\n");
    printf("  Sirali : %s\n", convex_sirali(cp, n) ? "Convex" : "Concave");
    printf("  Paralel: %s\n", convex_paralel(cp, n, 4) ? "Convex" : "Concave");

    printf("\n=== Test 2: Concave (beklenen: Concave) ===\n");
    printf("  Sirali : %s\n", convex_sirali(cc, n) ? "Convex" : "Concave");
    printf("  Paralel: %s\n\n", convex_paralel(cc, n, 4) ? "Convex" : "Concave");
}

/* ─────────────────────────────────────────────
 * Ana program
 * ───────────────────────────────────────────── */
int main() {
    int b;
    int boyutlar[7];
    int tekrar = 100;

    boyutlar[0] = 100;
    boyutlar[1] = 500;
    boyutlar[2] = 1000;
    boyutlar[3] = 5000;
    boyutlar[4] = 10000;
    boyutlar[5] = 50000;
    boyutlar[6] = 100000;

    printf("=========================================\n");
    printf(" Paralel Convex Polygon Kontrol Sistemi  \n");
    printf(" OpenMP - Grup 2                         \n");
    printf("=========================================\n\n");
    printf("Max kullanilabilir thread: %d\n\n", omp_get_max_threads());

    manuel_test();

    printf("=== Hizlanma Katsayisi Tablosu ===\n");
    printf("(Her olcum %d tekrarin ortalamasi)\n", tekrar);

    for (b = 0; b < 7; b++) {
        int n = boyutlar[b];
        Nokta *p = (Nokta*)malloc(n * sizeof(Nokta));
        if (!p) { fprintf(stderr, "Bellek hatasi!\n"); return 1; }
        duggun_polygon_olustur(p, n);
        hizlanma_tablosu(p, n, tekrar);
        free(p);
    }

    printf("\n=========================================\n");
    return 0;
}