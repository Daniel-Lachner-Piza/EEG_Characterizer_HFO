import cv2 as cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
 
"""
Object Detection, Contour detection
"""

def hfo_spectral_analysis(hfo_spect_img: np.ndarray = None, fs:int=None, wdw_duration_ms:int=None, cwt_range_Hz:tuple[int, int]=None, plot_ok:bool=False, fig_title:str='Title', out_path:str=None):

    plot_summary = False

    # Resize spectrogram, keep things uniform
    #the image size is assuming an wdw_analysis duration of 1 second (1000ms) and a wavelet transform from 60Hz to 500Hz (440Hz range)
    # Any changed to the image size will prevent easily extracting the features from the spectrogram image
    original_img_sample_length= hfo_spect_img.shape[1]
    img_resize_width = wdw_duration_ms
    img_resize_height = cwt_range_Hz[-1]-cwt_range_Hz[0]
    hfo_spect_img = cv2.resize(hfo_spect_img, (img_resize_width, img_resize_height), interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(hfo_spect_img)
    brg_extract = r
    bgr_img_for_cluster = np.transpose(np.vstack((b.ravel(), g.ravel(), r.ravel())))
  
    #kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=42).fit(bgr_img_for_cluster)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(bgr_img_for_cluster)
    best_cluster_labels = kmeans.labels_

    if plot_summary:
        clstrd_img = hfo_spect_img.copy()
        bgr_colors = [(255,0,0), (0, 255,0), (0,0,255)]
        for row_idx in range(clstrd_img.shape[0]):
            pred_cluster = kmeans.predict(clstrd_img[row_idx, :,:])
            clstrd_img[row_idx, pred_cluster==0,:] = (255,255,255)
            clstrd_img[row_idx, pred_cluster==1,:] = (127,127,127)
            clstrd_img[row_idx, pred_cluster==2,:] = (25, 25, 25)


    centroids = []
    for clstr_label in np.unique(best_cluster_labels):
        clstr_sel = best_cluster_labels == clstr_label
        centroid_val = np.mean(brg_extract.ravel()[clstr_sel])
        centroids.append(centroid_val)

    # Global Threshold hsv channel
    # th_val = np.percentile(hsv_img_extract_inv, 99, axis=None)
    #th_val = np.median(hsv_img_extract_inv)+5*np.std(hsv_img_extract_inv)
    th_val = int(np.max(centroids))
    ret, th_img = cv2.threshold(brg_extract, th_val, 255, cv2.THRESH_BINARY)


    # Get Contours
    contours, hierarchy = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plot Contours
    objects = []
    if plot_ok:
        objects = hfo_spect_img.copy()
        cont_thickness = 2
        cont_color = (128, 128, 128)
    
    all_contours = []
    rel_cont_idx = 0
    for contour in contours:
        
        area = cv2.contourArea(contour)
        if area < 1:
            continue
        
        perimeter = cv2.arcLength(contour, closed=True)
        circularity = 100*(4*np.pi*(area/(perimeter**2)))
        # Centroids
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        freq_centroid_Hz = cwt_range_Hz[-1]-cy
        freq_min_Hz = cwt_range_Hz[-1]-np.max(contour[:,:,1])
        freq_max_Hz = cwt_range_Hz[-1]-np.min(contour[:,:,1])
        vspread = freq_max_Hz-freq_min_Hz
        hspread = np.max(contour[:,:,0])-np.min(contour[:,:,0])
        hvr = 100*hspread/vspread

        correct_endpoints = True
        if correct_endpoints:
            contour_start_ms = np.min(contour[:,:,0])
            contour_end_ms = np.max(contour[:,:,0])
            contour_center_ms = cx
            contour_period_ms = 1000/freq_centroid_Hz
            
            #  if the blob is very small, extend its limits so that it fits 6 oscillations at its frequency
            # if contour_center_ms-(2*contour_period_ms) < contour_start_ms:
            #     contour_start_ms = contour_center_ms-(2*contour_period_ms)
            # if contour_center_ms+(2*contour_period_ms) > contour_end_ms:
            #     contour_end_ms = contour_center_ms+(2*contour_period_ms)

            contour_start_ms -= contour_period_ms
            contour_end_ms += contour_period_ms
            if contour_start_ms<0:
                contour_start_ms=0
            if contour_end_ms>img_resize_width:
                contour_end_ms=img_resize_width

            dur_ms = contour_end_ms-contour_start_ms
            nr_oscillations = np.round(dur_ms/contour_period_ms)
        else:
            contour_start_ms = np.min(contour[:,:,0])
            contour_end_ms = np.max(contour[:,:,0])
            contour_period_ms = 1000/freq_centroid_Hz
            contour_center_ms = np.mean([contour_start_ms, contour_end_ms])
            dur_ms = contour_end_ms-contour_start_ms
            hspread = dur_ms
            hvr = 100*hspread/vspread
            nr_oscillations = np.round(dur_ms/contour_period_ms)


        # if nr_oscillations <= 1:
        #     continue

        # if freq_centroid_Hz < 80:
        #     #continue
        #     pass

        valid_contour = True

        contour_features = defaultdict(list)
        contour_features['idx'] = rel_cont_idx
        contour_features['area'] = area
        contour_features['perimeter'] = perimeter
        contour_features['circularity'] = circularity
        contour_features['hspread'] = hspread
        contour_features['vspread'] = vspread
        contour_features['hvr'] = hvr
        contour_features['center_ms'] = contour_center_ms #cx
        contour_features['start_ms'] = contour_start_ms #np.min(contour[:,:,0])
        contour_features['end_ms'] = contour_end_ms #np.max(contour[:,:,0])
        contour_features['dur_ms'] = dur_ms
        contour_features['freq_centroid_Hz'] = freq_centroid_Hz
        contour_features['freq_min_Hz'] = freq_min_Hz
        contour_features['freq_max_Hz'] = freq_max_Hz
        contour_features['nr_oscillations'] = nr_oscillations
        contour_features['spect_ok'] = valid_contour
        all_contours.append(contour_features)

        if plot_ok:
            #metrics_str = f"Blob:{rel_cont_idx}, Freq:{freq_centroid_Hz}, NrOsc:{nr_oscillations}, Dur:{hspread:.0f}, VS:{vspread:.0f}, Circ:{circularity:.0f}, HVR:{hvr:.0f},Area:{area:.0f}"
            metrics_str = f"Blob:{rel_cont_idx}, Area:{area:.0f}, Freq:{freq_centroid_Hz}, NrOsc:{nr_oscillations}, Circ:{circularity:.0f}, HVR:{hvr:.0f}"

            # Show contour and centroid in image
            #if area < 300 or circularity < 0.3 or hvr < 20:
            metrics_str_color = (255, 255, 255)
            if valid_contour:
                cont_color = (255, 0, 255)
                metrics_str_color = (255, 0, 255)
            else:
                cont_color = (128, 128, 128)

            cv2.putText(img=objects, text=metrics_str, org=(0, 30+15*rel_cont_idx), fontFace=1, fontScale=1, color=metrics_str_color, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin = False) 

            cv2.drawContours(image=objects, contours=[contour], contourIdx=-1, color=cont_color, thickness=cont_thickness)
            cv2.circle(objects, (cx, cy), 4, (0,0,0), -1)

            # Add contour number
            font = cv2.FONT_HERSHEY_PLAIN  
            fontScale = 1
            txt_thickness = 1
            #annot_str = f"F:{freq_centroid_Hz:.0f}, A:{area:.0f}, C:{circularity:.0f}, HS:{hspread:.0f}, VS:{vspread:.0f}, HVR:{hvr:.0f}"
            #annot_str = f"A:{area:.0f}C:{100*circularity:.0f}, HVR:{hvr:.0f}"
            annot_str = f"{rel_cont_idx}"
            org = (int(cx), cy)
            cv2.putText(objects, annot_str, org, font, fontScale, (0, 0, 0), txt_thickness, cv2.LINE_AA, bottomLeftOrigin = False) 
        
        rel_cont_idx += 1

        pass

    save_img = False
    if save_img:
        objects = cv2.resize(objects, (860, 400), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_path+fig_title+'.png', objects)
    
    if plot_summary:
        new_size = (860, 400)
        split_img_a = np.concatenate((cv2.resize(hfo_spect_img, new_size, interpolation=cv2.INTER_LINEAR), \
                            cv2.resize(clstrd_img, new_size, interpolation=cv2.INTER_LINEAR)), axis=1)
        cv2.line(split_img_a, (new_size[0],0), (new_size[0],new_size[1]), (255,255,255), 1) 

        split_img_b = np.concatenate((cv2.resize(brg_extract, new_size, interpolation=cv2.INTER_LINEAR), \
                            cv2.resize(th_img, new_size, interpolation=cv2.INTER_LINEAR)), axis=1)
        cv2.line(split_img_b, (new_size[0],0), (new_size[0],new_size[1]), (255,255,255), 1) 

        split_img_c = np.concatenate((cv2.resize(hfo_spect_img, new_size, interpolation=cv2.INTER_LINEAR), \
                            cv2.resize(objects, new_size, interpolation=cv2.INTER_LINEAR)), axis=1)
        cv2.line(split_img_c, (new_size[0],0), (new_size[0],new_size[1]), (255,255,255), 1) 
        
        cv2.imshow("HFO Spectro vs Clustered Spectro"+"     "+fig_title, split_img_a)
        cv2.imshow("Red Channel vs Thresholded Red Channel"+"     "+fig_title, split_img_b)
        cv2.imshow("HFO Spectro vs Contours"+"     "+fig_title, split_img_c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass

    all_contours_df = pd.DataFrame(all_contours)  

    return objects, all_contours_df

if __name__ == "__main__":
    pass
